import torch
import mmap
import random
from vocab import encode
from hyperparams import train_file, val_file, block_size, batch_size, device

# Memory map for using small snippets of text from a single file of any size without having to open the whole file
def get_random_chunk(split):
    filename = train_file if split == "train" else val_file

    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequence
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data

def get_batch(split):
    data = get_random_chunk(split)
    
    # Get 4 (batch_size) random integers which represent the start indices for our blocks
    # (ensure that we have enough characters in sequence hence len(data) - block_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Get a tensor with 4 blocks
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Same blocks but then offset by 1
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # Send to GPU, meaning they will be processed in parallel
    x, y = x.to(device), y.to(device)
    
    return x, y