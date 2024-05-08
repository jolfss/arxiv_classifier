#!/home/pcgta/mambaforge/envs/arxiv/bin/python
# #!/home/sean/mambaforge/envs/arxiv/bin/python3
#----------------------------------#
#   Sean Brynjólfsson (smb459)     #
#   Deep Learning * Assignment 6   #
#----------------------------------#
"""This file contains the training script."""

#-------------#
#   imports   #
#-------------#
import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.tensorboard import SummaryWriter

# local import
from core import *
from data import *

#----------------------#
#   data preparation   #
#----------------------#
train_dataset = EmbeddingDataset(train_x, train_y, batch_size=32)
train_dataloader, validation_dataloader = split_dataloader(train_dataset, batch_size=256, split=0.9)

#-----------------------#
#   model preparation   #
#-----------------------#
# sumwriter = SummaryWriter()

# model = SimpleEmbeddingClassifier(768, 2048, 151)

criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(),lr=1e-3) 

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=False)

#-------------------#
#   training loop   #
#-------------------#

# basic training loop
# for epoch in (loop:=tqdm(range(100))):
#     train_loss = train(model, train_dataloader, optimizer, criterion, epoch, sumwriter)
#     val_loss, val_acc = validate(model, validation_dataloader, criterion, epoch, sumwriter)
#     loop.desc = F"Training (val={str(100*val_acc)[:6]}%)"
#     scheduler.step(val_loss)

# counter = 0
# for mixup in range(10):
#     train_dataloader, validation_dataloader = split_dataloader(train_dataset, batch_size=256, split=0.2)
#     for epoch in (loop:=tqdm(range(10))):
#         train_loss = train(model, train_dataloader, optimizer, criterion, counter, sumwriter)
#         val_loss, val_acc = validate(model, validation_dataloader, criterion, counter, sumwriter)
#         counter+=1
#         loop.desc = F"Training (val={str(100*val_acc)[:6]}%)"
#         scheduler.step(val_loss)

min_val_loss = 100
max_val_acc = 0
max_model = None
for runs in range(10):
    model = SimpleEmbeddingClassifier(768, 2048, 151)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3) 
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=False)
    train_dataloader, validation_dataloader = split_dataloader(train_dataset, batch_size=256, split=0.9)
    sumwriter = SummaryWriter()
    for epoch in (loop:=tqdm(range(75))):
        train_loss = train(model, train_dataloader, optimizer, criterion, epoch, sumwriter)
        val_loss, val_acc = validate(model, validation_dataloader, criterion, epoch, sumwriter)
        loop.desc = F"Training (val={str(100*val_acc)[:6]}%)"
        scheduler.step(val_loss)
    if val_loss < min_val_loss:
        max_model = model
        max_val_acc = val_acc

# make predictions

# validation_dataset = Dataset(embed_decoder_last_layer_mean(test_x))
# validation_dataloader = DataLoader(validation_dataset, batch_size=256, shuffle=False)



if input("Would you like to save this model? [Y/n]:") in ['y','Y']:
    torch.save(model.state_dict(), F"models/{input("Give the model a name:")+F"_{val_acc}"}")