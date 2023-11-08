# LLM Research doc

In this doc I'll put some findings after researching training my LLM model with different parameters. Monitoring things like GPU usage and the like.

## First LLM Generation results

After completing the tutorial, I generated my first text by the LLM. I had trained it for only 20 iterations using the following hyper params:

```py
batch_size = 32
block_size = 32
max_iters = 20
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_layer = 4
n_head = 4
dropout = 0.2
```

Loss after training: 3.2149083614349365

First convo:

```bash
Prompt:
kwek
Completion:
mﱑ 勹柸플裏aꂼanofeb🤷b𪜶𒉇n 𐜟鳌leins໓蚂s⎷i
Prompt:
你好
Completion:
roa痩 tAere t寺ewꄪoeanӮeaイotcthi𤇃b迂
Prompt:
how are you?
Completion:
e1sere 劢es 𐇦𐨟 dere a4箬wn axwyp🚠eb
```

It's not making a lot of sense at this point lol.

### Action steps

- Doing a first "proper" training with current parameters for around 10000 - 20000 iterations.
- Monitoring loss of training + val split during this training session
- Monitoring GPU usage (python tool?)
- Rechecking if output will start to make more sense at this point
