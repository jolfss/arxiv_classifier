#!/home/sean/mambaforge/envs/arxiv/bin/python3
#----------------------------------#
#   Sean Brynjólfsson (smb459)     #
#   Deep Learning * Assignment 6   #
#----------------------------------#
# generic
from typing import List
import pandas as pd
from tqdm import tqdm 
# ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# hf
from tokenizers import Tokenizer
from transformers import GPT2Tokenizer, GPT2Model
# personal
from arxiv_data import arxiv_train_dataloader, arxiv_title_dataloader, arxiv_val_dataloader

# set up gpt2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(".gpt2_tokenizer")
gpt2_model = GPT2Model.from_pretrained(".gpt2_model")

# gpus
device = "cuda" if torch.cuda.is_available() else "cpu"

# dataloaders
#train_loader = arxiv_train_dataloader
train_loader = arxiv_title_dataloader
val_loader = arxiv_val_dataloader

# model
class ArxivClassifier(nn.Module):
    def __init__(self, din, dhidden, dout):
        super().__init__()
        self.fc1 = nn.Linear(din, dhidden, device=device)
        self.fc2 = nn.Linear(dhidden, dout, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x= self.fc2(x)
        return x.squeeze(dim=1)

model = ArxivClassifier(768, 256, 151)

# training
sumwriter = SummaryWriter()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

def train():
    model.train()
    total_loss = 0
    for x,y in tqdm(train_loader):
        x,y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(train_loader)
    print(F"Loss: {avg_loss}")
    sumwriter.add_scalar('Loss (Train)', avg_loss, epoch)    
    return avg_loss

# validation
def validate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  
        for x, y in (val_loader):
            x,y = x.to(device), y.to(device)
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == y).sum().item()
            total_predictions += y.size(0)
    avg_loss = total_loss/len(val_loader)
    accuracy = correct_predictions / total_predictions
    print(f"Validation Loss:{avg_loss} Accuracy:{accuracy}")
    sumwriter.add_scalar('Loss (Validation)', avg_loss, epoch)
    sumwriter.add_scalar('Accuracy (Validation)', accuracy, epoch)
    return avg_loss, accuracy

# executable

num_epochs = 150  # Example number of epochs

for epoch in tqdm(range(num_epochs)):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train()
    val_loss, val_accuracy = validate()

