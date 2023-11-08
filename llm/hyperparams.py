import torch
# import argparse

# parser = argparse.ArgumentParser(description="This is a demonstration program")
# parser.add_argument("-batch_size", type=str, required=True, help="Please provide a batch_size")
# args = parser.parse_args()

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

# Folder/file hyperparams (running python inside of llm folder!)
vocab_file = "../train_data/vocab.txt"
train_file = "../train_data/train_split.txt"
val_file = "../train_data/val_split.txt"
pkl_file = "model-01.pkl"

# LLM Hyperparams

batch_size = 32 # How many sets of blocks can be evaluated at the same time (the more, the more gpu intensive)
block_size = 32 # Size of the blocks used to predict the next token
max_iters = 20 # Amount of iterations in training loop
learning_rate = 3e-4 # Experiment with this and find the one that has the best performance + quality over time
# Some good values to test out: 3e-3, 3e-4, 1e-3, 1e-4
eval_iters = 100
n_embd = 384 # Length of vector for each item in the vocab (amnt of info to store per letter)
n_layer = 4 # Amount of blocks/layers to run sequentually
n_head = 4 # Amount of heads to run in parallel in MultiHeadAttention
dropout = 0.2 # Drop 20% of our neurons to prevent overfitting