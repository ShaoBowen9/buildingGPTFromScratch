import torch

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text))) #all types of characters
vocab_size = len(chars) #amount of characters

stoi = { ch:i for i,ch in enumerate(chars)} #string to integer for every character
itos = { i:ch for i,ch in enumerate(chars)} #given an integer, return the corresponding character
encode = lambda s: [stoi[c] for c in s] #takes s as input, returns a list of integers using stoi
decode = lambda l: ''.join([itos[i] for i in l]) #for every number in l, add the corresponding character

''' test encoding/decoding
tmp = 'terry'
print(stoi)
print(encode(tmp))
print(decode(encode(tmp)))
'''

data = torch.tensor(encode(text), dtype =torch.long) #converts text to a tensor with 64-bit int values
"""
Note: 1D tensor as its just an array of characters, 64-bit specified in torch.long
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this
"""

n = int(0.9*len(data))
block_size = 8 #context length
train_data = data[:n] #90% used for training
val_data = data[n:] #10% used for validation

x = train_data[:block_size] #all context tensors
y = train_data[1:block_size+1] #all targets
for t in range(block_size): 
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target:{ target}")
