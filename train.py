import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 4
block_size = 8
max_iters = 3000
eval_interval = 300
learning_ate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#--------------
torch.manual_seed(1337)

#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
#all characters that appear in input.txt
chars = sorted(list(set(text))) #all types of characters
vocab_size = len(chars) #amount of characters

#maps characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars)} #string to integer for every character
itos = { i:ch for i,ch in enumerate(chars)} #given an integer, return the corresponding character
encode = lambda s: [stoi[c] for c in s] #takes s as input, returns a list of integers using stoi
decode = lambda l: ''.join([itos[i] for i in l]) #for every number in l, add the corresponding character

#data loading, generates batch of inputs at random
def get_batch(split):
    data = train_data if split == 'train'else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

data = torch.tensor(encode(text), dtype =torch.long) #converts text to a tensor with 64-bit int values

''' test encoding/decoding
tmp = 'terry'
print(stoi)
print(encode(tmp))
print(decode(encode(tmp)))
'''

"""
Note: 1D tensor as its just an array of characters, 64-bit specified in torch.long
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this
"""

n = int(0.9*len(data))
block_size = 8 #context length
train_data = data[:n] #90% used for training
val_data = data[n:] #10% used for validation

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets')
print(yb.shape)
print(yb)

print('--------')
for b in range(batch_size): #batch dimenion
    for t in range(block_size): #time dimension/each character
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")