"""
- Tokenized text for LLM traning is "The Verdict"
Wiki: https://en.wikisource.org/wiki/The_Verdict

Github repositoy for "the-verdict.txt": @ https://mng.bz.Adng
 or download with code below
 """

import urllib.request
import re 
from pathlib import Path

# 1. download if not already present 
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = Path("the-verdict.txt")
if not file_path.exists():
    urllib.request.urlretrieve(url, file_path)

# 2. Load raw text
raw_text = file_path.read_text(encoding="utf-8")
print("Length of raw text:", len(raw_text))
print ("First 99 chars =>", raw_text[:99])

# 3. Quick regex demo - Start of Preprocessing
sample = "Hello, world. This, is a test."
print(re.split(r'\s', sample)) # Splits on spaces 
print(re.split(r'([,\.]|\s)', sample)) # Splits on spaces and [, .] punctuation as shown in split 

# 4. prepare full token list
tokens = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# Why above |--|? Because in the text there are some dashes that are not part of a word, but rather used to separate words.
tokens = [t.strip() for t in tokens if t.strip()] # remove empty strings
vocab = {tok: idx for idx, tok in enumerate(sorted(set(tokens)))}
print("Vocab size:", len(vocab))

# 5. Create tokenizer classes
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text: str):
        pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text) # same as above: split by spaces and punctuation
        pieces = [p.strip() for p in pieces if p.strip()] # remove empty strings
        return [self.str_to_int[p] for p in pieces]
    
    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text) # remove space before punctuation and substitute with \1 because \1 is the first group in the regex
    
# 6. quick round-trip test
tok = SimpleTokenizerV1(vocab)
demo = ('"It\'s the last he painted, you know," '
        'Mrs. Gisburn said with pardonable pride.')
demo_ids = tok.encode(demo)
print("Encoded ID:", demo_ids[:10], "...")
print("Decoded =>", tok.decode(demo_ids))

# 7. Unknown-token aware tokenizer
UNK, EOS = "<|unk|>", "<|endoftext|>"
full_vocab = sorted(set(tokens) | {UNK, EOS})
v2_vocab = {tok: idx for idx, tok in enumerate(full_vocab)}

class SimpleTokenizerV2(SimpleTokenizerV2 := object):
    def __init__(self, vocab):
        self.str_to_int = vocab 
        self.int_to_str = {i: s for s, i in vocab.items()} # reverse vocab from int -> string via i:s (key:value)

    def encode(self, text):
        pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text) # same as above: split by spaces and punctuation
        pieces = [p.strip() for p in pieces if p.strip()] # removes empty strings
        pieces = [p if p in self.str_to_int else UNK for p in pieces] # replaces unknown tokens with UNK and keeps the rest
        return [self.str_to_int[p] for p in pieces]
    
    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text) # remove space before punctuation and substitute with \1 because \1 is the first group in the regex
    
tok2 = SimpleTokenizerV2(v2_vocab)
text1 = "Hello do you like tea?"
text2 = "In the sunlit terraces of the palace."
both = f"{text1} {EOS} {text2}"
print("Encoded with UNK:", tok2.encode(both))
print("Decoded with UNK:", tok2.decode(tok2.encode(both)))

"""
Additional special tokens:
- <|pad|> for padding
- <|BOS|> for beginning of sequence
- <|eos|> for end of sequence
- <|unk|> for unknown tokens
- <|mask|> for masked tokens
- <|cls|> for classification
- <|sep|> for separating sequences
- <|unused0|> to <|unused99|> for unused tokens
- <|sop|> for start of paragraph
- <|eop|> for end of paragraph
- <|sos|> for start of sentence
- <|eos|> for end of sentence
- <|sop|> for start of object
- <|eop|> for end of object
"""
# GPT Models only use <|EOS|> 
# Also GPT models use a byte pair encodfing tokenizer (BPE)


# BPE Tokenizer
# Tiktoken is a library for BPE tokenization - https://github.com/openai/tiktoken
# BPE is a subword tokenization algorithm that is used to encode text into tokens.
from importlib.metadata import version
import tiktoken
print("Tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces" 
    "of someunknownPlace."
)
# Returns a list of integers representing the tokens (integers:string)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# Decode the integers back to text
strings = tokenizer.decode(integers)
print(strings)

# Excercise: 2.1 BByte pair encoding of unknown words
text = "Akwirw ier"
token_ids = tokenizer.encode(text)

# Raw Token-ID list
print()
print(token_ids)

# Decode the token IDs back to text piece
# - tokenizer.decode() expects a list of token IDs
# - repr() show leading spaces/newlines explicitly
for token_id in token_ids:
    piece = tokenizer.decode([token_id])
    print(f"{token_id:>5} -> {repr(piece)}")
print()

# Decode the whole list to check round-trip
reconst = tokenizer.decode(token_ids)
print(repr(reconst))
assert reconst == text, "Round-trip failed!"

print()
print()

# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# Number fo tokens in training set
enc_text = tokenizer.encode(raw_text)
print("Length of raw text:", len(raw_text))


enc_sample = enc_text[50:] # 50 tokens
print("Encoded sample:", enc_sample)

# Next - Word prediciton task
context_size = 4 # Context size determines the amount of tokens for input
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1] # shifted by one token
print(f"x: {x}")
print(f"y: {y}")
# x is the input and y is the target

# To process
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
# Left of the arrow is the inputs and the right side is the LLM Supposed to predict
print()
# Convert token IDs into text but similiar to above 
for i in range(1, context_size + 1) :
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# A dataset for batched inputs and targets
import torch
from torch.utils.data import Dataset, DataLoader

# Dataset class
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) # Tokenize the entire text
        # USES A sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Returns the total number of rows in the dataset.
    def __len__(self):
        return len(self.input_ids)
    
    # Returns a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
# DataLoader 
# A data loader to generate batches with input-with pairs
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2") # Initialize the tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) # Create the dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, # Drop_last = True; drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training
        num_workers=num_workers, # The number of CPU processes to use for preprocessing
    )
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) # Converts dataloader into a python iterator to fetch the next entry via Python's built-in next() function
first_batch = next(data_iter)
# First_batch variable contains two tensors: the first tensor stores the input token IDs
# Second tensor stores the target token IDs. # Max length indicates the number of tokens in each tensor (common: 256)
print(first_batch)
print()
second_batch = next(data_iter)
# The stride paramter dictates the number of positions the inputs shift across batches, emulating a sliding window approach

print(second_batch)
print()

# Excersice 2.2 Data Loaders with different strides and context sizes
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=2, stride=2, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
print()
second_batch = next(data_iter)
print(second_batch)
print()
# or
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
print()
second_batch = next(data_iter)
print(second_batch)
print()
# DataLoader with a batch size > 1
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("inputs:\n", inputs)
print("\ntargets:\n", targets)
print()

# Increase overlap between the batches could lead to increased overfitting which is why our stride = max_length

# How the token ID to embedding vector conversion works 
input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3 # embedding dimension
torch.manual_seed(123) # set the random seed for reproducibility
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight) # One row for each of the six possible tokens and one column for each of the three embedding dimensions
print()
print(embedding_layer(torch.tensor([3]))) # Look up operation that retrieves the embedding vector for the token ID 3
# Above is an exmaple of one-hot encoding: https://mng.bz/ZEB5
print(embedding_layer(input_ids)) # Look up operation that retrieves the embedding vectors for the token IDs 2, 3, 5, and 1

# Relative position embeddings and absolute position embeddings help the model understand the order of the tokens in the input sequence
# Absolute position embeddings are fixed and added to the token embeddings
# Relative position embeddings are learned during training and are used to capture the relationships between tokens in the input sequence 
print()

new_vocab_size = 50257
new_output_dim = 256
token_embedding_layer = torch.nn.Embedding(new_vocab_size, new_output_dim)
# The token embedding layer is initialized with random weights
# Batch size of 8 and our max length is 4 and 256 dim - batch_size * max_length * embedding_dim
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:", inputs.shape)
# Now embed 256 dimensional vectors
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
print()
context_length = max_length 
pos_embedding_layer = torch.nn.Embedding(context_length, new_output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print("Positional Embeddings:\n", pos_embeddings)
print("\nPositional Embeddings shape:", pos_embeddings.shape)
input_embededdings = token_embeddings + pos_embeddings
print(input_embededdings.shape)
print()
print()
# Attention Mechanisms: Simplified self-attention, self-attention, casual attention, and Multi-head attention

# RNNs use the hidden state to capture the context of the input sequence and is a type
# of neural network where outputs form previous steps are fed as inputs to the current step
# context vector interpreted  as an enriched embedding vector to represent the input sequence

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.33, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55], # step (x^6)
     ]
)
print(inputs.shape)
print(inputs)
print()

query = inputs[1] # The second input token serves as the query
attn_scores_2 = torch.empty(inputs.shape[0]) # Starts with an empty tensor then utilizes the dot product to calculate the attention scores
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # The dot product between the query and each input token
print("Attention scores for the second token:\n", attn_scores_2)
print()

# Refresh dot product
res = 0
 # Takes each element from first input token [0.43, 0.15, 0.89] and query is second input token [0.55, 0.87, 0.66]
 # Then takes first element 0.43 * 0.55, second element 0.15 * 0.87, and third element 0.89 * 0.66
 # Then sums them up to get the final result which is (0.2365 + 0.1305 + 0.5874) = 0.9544
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print(res)
print(torch.dot(inputs[0], query)) # The dot product between the first input token and the query
# Measure of similiarty between two vectors: a higher dot product indicates a greater degree of alignment or similiarity between the vectors.
# Thus a higher attention score indicates a greater degree of alignment or similiarity between the query and the input token
print()
# Normalize the attention scores
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum", attn_weights_2_tmp.sum())
print()
# Better alternative is to use softmax function
attn_weights_2 = torch.nn.functional.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)

# Basic implementation of softmax funciton for normalizing the attention scores
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# OR 

print()
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
print()

# Calculate one context vector

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("Context vector for the second token:\n", context_vec_2)
print()

# Calculate context vector for all tokens
attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print("Attention scores:\n", attn_scores)
print()

"""
Steps:
 1) Compute the attention scores as dot products between the inputs
 2) The attention weights are a normalized version of the attention scores
 3) The context vectors are computed as a weighted sum over the inputs
 4) The context vectors are then used as the input to the next layer of the network
"""

# Calculate attention scores for all tokens instad with matrix multiplacation using the @ operator (matrix multiplication)
attn_scores = inputs @ inputs.T # The inputs.t() transposes the inputs tensor
print("Attention scores:\n", attn_scores)
print()
# Normalize the attention scores
attn_weights = torch.softmax(attn_scores, dim=-1) 
print("Attention weights:\n", attn_weights)
print()


# Verify rows sum to 1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))
print()

# Calculate context vectors for all tokens
all_context_vecs = attn_weights @ inputs
print("Context vectors:\n", all_context_vecs)
print()
print("Previous 2n context vector:\n", context_vec_2)

# Self Attention three trainable weight matrices: W_q, W_k, and W_v
# W_q is used to transform the input into query vectors
# W_k is used to transform the input into key vectors
# W_v is used to transform the input into value vectors
# The attention scores are computed as the dot product of the query and key vectors
# The attention weights are computed as the softmax of the attention scores
# The context vectors are computed as the weighted sum of the value vectors

x_2 = inputs[1] # The second input element
d_in = inputs.shape[1] # The input embedding size,d = 3
d_out = 2 # The output embedding size, d_out = 2

# INIT 
torch.manual_seed(123) # Set the random seed for reproducibility
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
print("W_query:\n", W_query)
print("W_key:\n", W_key)
print("W_value:\n", W_value)

# Compute the query, key, and value vectors
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value 
print(query_2)
print()
print(key_2)
print() 
print(value_2)

keys = inputs @ W_key
values = inputs @ W_value
print("Keys:\n", keys.shape)
print("Values.shape:\n", values.shape)
print("Keys:\n", keys)
print()

# Computing the attention score of W(22):
keys_2 = keys[1]
attn_scores_2 = query_2.dot(keys_2)
print("Attention scores for the second token:\n", attn_scores_2)
print()

# Computing attention score of W(2T):
attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print("Attention scores for the second token:\n", attn_scores_2)
print()

# Square root same as exponentiation (0.05)
d_k = keys.shape[-1] # Last dimension - embedding size
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # It normalizes the attention scores by dividing them by the square root of the dimension of the key vectors
print("Attention weights for the second token:\n", attn_weights_2)
print()