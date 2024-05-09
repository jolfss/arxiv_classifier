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

# criterion = nn.CrossEntropyLoss()

# max_length = 512 

# train_dataset = EmbeddingDataset(train_x, train_y, to_raw_embeddings, batch_size=512)

# train_dataloader = DataLoader(train_dataset, shuffle=True)

# sumwriter = SummaryWriter()
# train_dataloader, validation_dataloader = split_dataloader(train_dataset, batch_size=4096, split=0.95)
# model = SimpleEmbeddingClassifier(768, 2048, 151)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

# for epoch in (loop := tqdm(range(50))):
#     train_loss = train(model, train_dataloader, optimizer, criterion, epoch, sumwriter)
#     val_loss, val_acc = validate(model, validation_dataloader, criterion, epoch, sumwriter)
#     loop.desc = f"Training (val={str(100 * val_acc)[:6]}%)"
#     scheduler.step(val_loss)


# max_val_acc = 0.56
# max_model = None

# criterion = nn.CrossEntropyLoss()
# train_dataset = EmbeddingDataset(train_x, train_y, embedder=to_raw_embeddings, partial_credit=True, batch_size=32)

# for run in range(10):
#     sumwriter = SummaryWriter(comment=f"Run #{run}")
#     train_dataloader, validation_dataloader = split_dataloader(train_dataset, batch_size=2*4096, split=0.8)
#     model = SimpleEmbeddingClassifier(768, 2048, 151)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)
    
#     for epoch in (loop := tqdm(range(50))):
#         train_loss = train(model, train_dataloader, optimizer, criterion, epoch, sumwriter)
#         val_loss, val_acc = validate(model, validation_dataloader, criterion, epoch, sumwriter)
#         loop.desc = f"Training (val={str(100 * val_acc)[:6]}%)"
#         scheduler.step(val_loss)

#         if val_acc > max_val_acc:
#             max_val_acc = val_acc
#             torch.save(model.state_dict(), "best_model.pt")
#             #print(f"New best model saved with validation accuracy: {max_val_acc}")

# if input(f"Would you like to save the best model ({max_val_acc})? [Y/n]: ").lower() == 'y':
#     model_name = input("Give the model a name: ")
#     torch.save(model.state_dict(), f"models/{model_name}_{max_val_acc}.pt")
#     print(f"Model saved as {model_name}_{max_val_acc}.pt")

# make predictions

validation_dataset = EvaluationDataset(test_x, embedder=to_raw_embeddings, name="evaluation",batch_size=32)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

best_model = SimpleEmbeddingClassifier(768, 2048, 151)
best_model.load_state_dict(torch.load("models/specter_0.5780567307025453.pt", map_location=device))
best_model.to(device)
preds = [pred.item() for pred_batch in evaluate(best_model, validation_dataloader) for pred in pred_batch.cpu()]

save_predictions(list(test_x['id']), preds)
