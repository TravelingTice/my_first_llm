import torch
from torch import nn
from torch.nn import functional as F
from hyperparams import n_embd, dropout, block_size, n_layer, n_head, device, pkl_file
from vocab import vocab_size
import pickle

class Head(nn.Module):
	""" one head of self-attention """
	
	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		# Registering tril ahead of time to reduce training time + costs and increase performance (reduce overhead by initting beforehand)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# input of size (batch, time-step (block size?), channels)
		# output of size (batch, time-step, head size)
		B,T,C = x.shape
		k = self.key(x) # B,T,hs
		q = self.query(x) # B,T,hs
		# compute attention scores ("affinities")
		# Scaling by 1/sqrt(length of row in keys/queries matrix) to prevent exploding
		weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) DOT PRODUCT (B, hs, T) -> (B, T, T) -> Apply the scaling
		# Apply a masked_fill to prevent looking into the future
		weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
		# Apply softmax
		weights = F.softmax(weights, dim=-1) # B, T, T
		# Apply dropout
		weights = self.dropout(weights)
		# Perform the weighted aggregation of the values, matrix multiply with value for weights!
		value = self.value(x) # B, T, hs
		out = weights @ value # B, T, T @ B, T, hs -> B, T, hs
		return out
		

class MultiHeadAttention(nn.Module):
	""" Multiple heads of self-attention in parallel """

	def __init__(self, num_heads, head_size):
		super().__init__()
		# Heads that will be running in parallel in isolation
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		# Add more learnable params to help it learn more about the text
		self.projection = nn.Linear(head_size * num_heads, n_embd)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# Align heads by (B, T, C) C dimension (feature dimension)
		
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.projection(out))
		return out

class FeedForward(nn.Module):
	""" A simple linear layer followed by a non-linearity """

	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.ReLU(),
			nn.Linear(4 * n_embd, n_embd),
			nn.Dropout(dropout),
		)

	def forward(self, x):
		return self.net(x)

# Decoder block
class Block(nn.Module):
	""" Transformer block: communication followed by computation """

	def __init__(self, n_embd, n_head):
		# n_embd: embedding dimension, n_head: the number of heads we'd like
		super().__init__()
		# Number of features (n_embd) each head will capture in our multihead attention
		head_size = n_embd // n_head
		self.self_attention = MultiHeadAttention(n_head, head_size)
		self.feed_forward = FeedForward(n_embd)
		self.layer_norm1 = nn.LayerNorm(n_embd)
		self.layer_norm2 = nn.LayerNorm(n_embd)

	def forward(self, x):
		y = self.self_attention(x)
		x = self.layer_norm1(x + y)
		y = self.feed_forward(x)
		x = self.layer_norm2(x + y)
		return x

class GPTLanguageModel(nn.Module):
	def __init__(self, vocab_size):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.positioning_embedding_table = nn.Embedding(block_size, n_embd)

		# Decoder blocks/layers
		self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

		# Create a layernorm after processing of the decoder blocks
		self.layernorm_final = nn.LayerNorm(n_embd)
		self.languagemodelling_head = nn.Linear(n_embd, vocab_size)

		self.apply(self._init_weights)

	# Helps initialize our weights properly, no need to understand what is going on, just know it will make learning better
	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # std = Standard deviation
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		
		
	def forward(self, index, targets=None):
		B, T = index.shape
		
		# index and targets are both (B,T) tensor of integers
		tok_emb = self.token_embedding_table(index) # (B,T,C)
		pos_emb = self.positioning_embedding_table(torch.arange(T, device=device)) # (T,C)
		x = tok_emb + pos_emb
		
		# Feed the token embeddings + position encodings into our network!
		x = self.blocks(x) # (B,T,C)

		# Finalize + get the probabilities
		x = self.layernorm_final(x) # (B,T,C)
		logits = self.languagemodelling_head(x) # (B, T, vocab_size)
		
		if targets is None:
			loss = None
		else:
			# Batch (size), Time (sequence of integers we know), Channel (vocab size)
			B, T, C = logits.shape # Unpack dimensions
			# B & T not as important here, so we blend them together, as long as logits + targets have same batch
			# and time, we should be fine. We are more focused on vocab size (C) here
			# What we're doing here is change the shape of the tensor to the way cross_entropy expects, which is B, C, T rather than B, T, C
			logits = logits.view(B*T, C) # Reshape dimensions
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)
		
		return logits, loss
	
	def generate(self, index, max_new_tokens):
		# Index is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# Crop index to block size
			index = index[:, -block_size:]
			# get the predictions
			logits, loss = self.forward(index)
			# focus only on the last time step
			logits = logits[:, -1, :] # becomes (B, C) as we only care about the last token, hence -1 in the 2nd (T) dimension
			# apply softmax to get probabilities
			probs = F.softmax(logits, dim=-1) # (B, C)
			# sample from distribution
			index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
			# append sampled index to the running sequence
			index = torch.cat((index, index_next), dim=1) # (B, T+1)
		return index

model = GPTLanguageModel(vocab_size)

# Load saved model from pickle file
try:
	print("loading model parameters...")
	with open(pkl_file, "rb") as f:
		model = pickle.load(f)
	
	print("loaded successfully!")
except:
	print("model parameters not found, initializing new model")
	
model = model.to(device)