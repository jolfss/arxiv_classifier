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
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.tensorboard import SummaryWriter

# local import
from core import *
from data import *


max_val_acc = 0.57
max_model = None
max_model_params = ""

# lr=0.75e-3
# drop=0.3
criterion = nn.CrossEntropyLoss()

def sample_hyperparameters(lr, hdim, drop, batch, factor, partial):
    # new_lr = lr * 10**(np.random.uniform(-0.1, 0.1))
    # new_hdim = int(hdim * np.random.uniform(0.9, 1.1))
    # new_drop = np.clip(drop + np.random.uniform(-0.1, 0.1), 0, 1)
    # new_batch = int(2**np.round(np.log2(batch * np.random.uniform(0.9, 1.1))))
    # new_factor = np.clip(factor + np.random.uniform(-0.02, 0.02), 0.05, 0.5)
    # new_partial = partial * 10**(np.random.uniform(-0.15, 0.15))
    #return new_lr, new_hdim, new_drop, new_batch, new_factor, new_partial
    return 
max_hyperstring=None

lr=8e-4
hdim=2**12 + 2**11
drop=0.7
batch=2**12
factor=0.1
partial=0.05

# lr=8e-4
# hdim=2**12
# drop=0.55
# batch=2**13
# factor=0.1
# partial=0.03

# for i in range(25):
#     print(F"Iteration {i}")
#     #train_dataset = EmbeddingDataset(train_x, train_y, embedder=to_raw_embeddings, partial_credit=partial, batch_size=32)
#     #hyperparameters = sample_hyperparameters(lr=9e-4,hdim=4096,drop=0.5,batch=2**13,factor=0.1,partial=0.03)
#     hyperstring = F"lr{lr}_hdim{hdim}_drop{drop}_batch{batch}_factor{factor}_partial{partial}"
#     #print(F"Current parameters: {hyperstring}")
#     sumwriter = SummaryWriter(comment=hyperstring)
#     train_dataset = JointEmbeddingDataset(".allenai_specter2_aug2023refresh_base_None_embeddings.pt",
#                                           ".gpt2_last_layer_mean_embeddings.pt",
#                                           ".gpt2-xl_last_layer_mean_embeddings.pt", 
#                                           train_y, partial_credit=partial, batch_size=32)
#     train_dataloader, validation_dataloader = split_dataloader(train_dataset, batch_size=batch, split=0.95) 
#     model = SimpleEmbeddingClassifier(768 + 768 + 1600, hdim, 151, drop)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=factor) 
    
#     for epoch in (loop := tqdm(range(60))):
#         train_loss = train(model, train_dataloader, optimizer, criterion, epoch, sumwriter)
#         val_loss, val_acc = validate(model, validation_dataloader, criterion, epoch, sumwriter)
#         scheduler.step(val_loss)
#         loop.desc = f"Training (val={str(100 * val_acc)[:6]}%)"

#         # if epoch == 38:
#         #     for g in optimizer.param_groups:
#         #         g['lr'] = lr/5

#         if val_acc > max_val_acc: # Save best model across all
#             max_val_acc = val_acc
#             max_model = model
#             max_hyperstring= hyperstring
#     if val_acc > 0.6125: # Save models with sufficiently good validation anyway
#         torch.save(model.state_dict(), F"models/fat_90_slow_{hyperstring}_{val_acc}.pt")

# print(F"The maximum model had parameters: {max_hyperstring}")
# torch.save(max_model.state_dict(), f"fat_best_90_slow_{max_hyperstring}_{max_val_acc}.pt")
# print(f"Model saved as {model_name}_{max_val_acc}.pt")

# max_val_acc = 0.61
# max_model = None

# lr=0.75e-3
# criterion = nn.CrossEntropyLoss()
# train_dataset = EmbeddingDataset(train_x, train_y, embedder=to_raw_embeddings, partial_credit=False, batch_size=32)

# for run in range(30):
#     sumwriter = SummaryWriter(comment=f"Run")
#     train_dataloader, validation_dataloader = split_dataloader(train_dataset, batch_size=2**14, split=0.85) # 2**13
#     model = SimpleEmbeddingClassifier(768, 3072, 151, 0.3)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.1) 
    
#     for epoch in (loop := tqdm(range(90))):
#         train_loss = train(model, train_dataloader, optimizer, criterion, epoch, sumwriter)
#         val_loss, val_acc = validate(model, validation_dataloader, criterion, epoch, sumwriter)
#         scheduler.step(val_loss)
#         loop.desc = f"Training (val={str(100 * val_acc)[:6]}%)"

#         if val_acc > max_val_acc:
#             max_val_acc = val_acc
#             torch.save(model.state_dict(), "best_model.pt")
#             #print(f"New best model saved with validation accuracy: {max_val_acc}")

# if input(f"Would you like to save the best model ({max_val_acc})? [Y/n]: ").lower() == 'y':
#     model_name = input("Give the model a name: ")
#     torch.save(model.state_dict(), f"models/{model_name}_{max_val_acc}.pt")
#     print(f"Model saved as {model_name}_{max_val_acc}.pt")

# make predictions

EvaluationDataset(test_x, to_decoder_last_layer_mean_embeddings, name="evaluation")

validation_dataset = JointEvaluationDataset(".allenai_specter2_aug2023refresh_base_evaluation_embeddings.pt",".gpt2-xl_last_layer_mean_evaluation_embeddings.pt", name="evaluation",batch_size=32)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

best_model = SimpleEmbeddingClassifier(768 + 768 + 1600, 4096, 151, 0.3)
best_model.load_state_dict(torch.load("fat_best_90_slow_lr0.0005_hdim4096_drop0.55_batch4096_factor0.1_partial0.03_0.6332871012482663.pt", map_location=device))
best_model.to(device)
preds = [pred.item() for pred_batch in evaluate(best_model, validation_dataloader) for pred in pred_batch.cpu()]

save_predictions(list(test_x['id']), preds)
