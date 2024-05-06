#!/home/sean/mambaforge/envs/arxiv/bin/python3
# generic imports
from typing import List
import pandas as pd
from tqdm import tqdm
import gc
# ml modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# huggingface
from transformers import GPT2Tokenizer, GPT2Model

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-----------------#
#   huggingface   #
#-----------------#
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('.gpt2_tokenizer')
gpt2 = GPT2Model.from_pretrained('.gpt2_model') # NOTE: set to gpt2 if weights get lost
gpt2.to(device)
# gpt2_tokenizer.save_pretrained(".gpt2tokenizer")
# gpt2.save_pretrained(".gpt2model")

#---------------------------------#
#   special notes and resources   #
#---------------------------------#
# https://arxiv.org/category_taxonomy

train_x : pd.DataFrame = pd.read_csv('train_x.csv')
"""[title]: a string containing the title of a paper
[abstract]: a string containing the abstract of a paper
[id]: an integer storing the id of the paper
"""
train_y = pd.read_csv('train_y.csv')
"""[id]: an integer storing the id of the paper
[primary_subfield]: a string containing the abstract of a paper
[all_subfields]: a string containing space-separated subfield names
[label]: the numeric representation of the primary subfield
[all_subfields_numeric]: a string of space-separated numeric representation of all_subfields (ordered?)
"""
test_x = pd.read_csv('test_x.csv')
"""[title]: a string containing the title of a paper
[abstract]: a string containing the abstract of a paper
[id]: an integer storing the id of the paper
"""

# # Prompt version
# def to_prompts(x : pd.DataFrame) -> List[str]:
#     prompts = []
#     for title, abstract, id in x.iloc():
#         prompt = F"{title}\n{abstract}"
#         prompts.append(prompt)
#     return prompts

# def to_x(x: List[str]) -> List[torch.Tensor]:
#     embeddings = []
#     for prompt in tqdm(x, desc="Processing prompts"):
#         inputs = gpt2_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = gpt2(**inputs)
#         embeddings.append(outputs.last_hidden_state.mean(dim=1))
#     return embeddings

# embeddings

def fat_embed_titles(x: pd.DataFrame, batch_size: int = 32) -> List[torch.Tensor]:
    all_embeddings = []
    for start_idx in tqdm(range(0, len(x), batch_size)):
        end_idx = start_idx + batch_size
        batch = x.iloc[start_idx:end_idx]
        titles = batch.iloc[:, 0].tolist()
        inputs = gpt2_tokenizer(titles, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = gpt2(**inputs)
        embeddings = outputs.last_hidden_state
        all_embeddings.append(embeddings.cpu())

        del inputs, outputs, embeddings # force-free memory to hopefully increase throughput
        torch.cuda.empty_cache()
        gc.collect()
    return all_embeddings

def fat_embed_abstracts(x: pd.DataFrame, batch_size: int = 32) -> List[torch.Tensor]:
    all_embeddings = []
    for start_idx in tqdm(range(0, len(x), batch_size)):
        end_idx = start_idx + batch_size
        batch = x.iloc[start_idx:end_idx]
        abstracts = batch.iloc[:, 1].tolist()
        inputs = gpt2_tokenizer(abstracts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = gpt2(**inputs)
        embeddings = outputs.last_hidden_state
        all_embeddings.append(embeddings.cpu())

        del inputs, outputs, embeddings # force-free memory to hopefully increase throughput
        torch.cuda.empty_cache()
        gc.collect()
    return all_embeddings

def to_y(y: pd.DataFrame) -> List[torch.Tensor]:
    return [torch.tensor(label, dtype=torch.long) for label in y['label']]

### SAVING
gpt2.resize_token_embeddings(len(gpt2_tokenizer)) # Cursed... see https://github.com/huggingface/transformers/issues/487

# torch.save(fat_embed_titles(train_x, 256),".x_train_fat_titles.pt")
# torch.save(fat_embed_abstracts(train_x,16), ".x_train_fat_abstracts.pt") # NOTE: Too fat to fit in RAM.

#DATA
class ArxivDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TitleDataset(Dataset):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
    def __len__(self):
        return self.batch_size*(len(self.x)-1) + self.x[-1].size(0)

    def __getitem__(self, idx):
        return self.x[idx // self.batch_size][idx % self.batch_size][-1], self.y[idx]

arxiv_dataset = ArxivDataset(torch.load(".train_x_embeddings.pt"), to_y(train_y))
title_dataset = TitleDataset(torch.load(".train_x_fat_titles.pt"), to_y(train_y), 256)

split=0.2
val_size = int(split*len(arxiv_dataset))  
train_size = len(arxiv_dataset) - val_size

train_dataset, validation_dataset = random_split(arxiv_dataset, [train_size, val_size])

arxiv_train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
arxiv_title_dataloader = DataLoader(title_dataset, batch_size=64, shuffle=True)
arxiv_val_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)