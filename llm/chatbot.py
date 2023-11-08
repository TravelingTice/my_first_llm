import torch
from vocab import encode, decode
from hyperparams import device
from model import model

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f"Completion:\n{generated_chars}")
