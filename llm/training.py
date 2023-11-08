import torch
import pickle
from tqdm import tqdm
from model import model
from hyperparams import eval_iters, max_iters, learning_rate, pkl_file
from get_batch import get_batch

# LOSS ESTIMATION DURING TRAINING (STATS)

# PyTorch will not use gradients in this block (reduce computation costs)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# TRAINING LOOP

# Create a PyTorch optimizer
# AdamW = optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Training in progress...")
for iter in tqdm(range(max_iters)):
    # Current iteration nth of eval_iters
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, validation loss: {losses['val']:.3f}")
    
    # sample batch data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    # Zero gradient.... not sure about this
    optimizer.zero_grad(set_to_none=True)
    
    # And don't fully understand the next 2 but apparently it's part of the learning cycle
    loss.backward()
    optimizer.step()
print("Training complete!")
print(loss.item())

with open(pkl_file, 'wb') as f:
    pickle.dump(model, f)
print("model saved")