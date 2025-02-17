#!/home/pcgta/mambaforge/envs/arxiv/bin/python
# #!/home/sean/mambaforge/envs/arxiv/bin/python3
#----------------------------------#
#   Sean Brynjólfsson (smb459)     #
#   Deep Learning * Assignment 6   #
#----------------------------------#
"""Module Comment."""

#--------------------#
#   imports/device   #
#--------------------#
# generic imports
import os
import gc
import pandas as pd
from typing import Callable, List, Optional
from tqdm import tqdm
# ml modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# huggingface
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2Model, GPT2Tokenizer
# TODO: Pip install adapters
from adapters import AutoAdapterModel
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-----------------------#
#   pretrained models   #
#-----------------------#

load_pretrained = True
if load_pretrained:
    model_name = "gpt2"
    print(F"Loading model {model_name}...")
    safe_model_name = model_name.replace("/","_")
    match model_name:
        case "gpt2" | "gpt2-xl":
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(F"pretrained/{safe_model_name}_tokenizer",force_download=False)
                _model = GPT2Model.from_pretrained(F"pretrained/{safe_model_name}_model",force_download=False)
            except: # fetch model if not found
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                _model = GPT2Model.from_pretrained(model_name)
                # save for future use
                tokenizer.save_pretrained(F"pretrained/{safe_model_name}_tokenizer")
                _model.save_pretrained(F"pretrained/{safe_model_name}_model")
        case "facebook/bart-large-mnli":
            try:
                tokenizer = AutoTokenizer.from_pretrained(F"pretrained/{safe_model_name}_tokenizer")
                _model = AutoModelForSequenceClassification.from_pretrained(F"pretrained/{safe_model_name}_model")
            except:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                _model = AutoModelForSequenceClassification.from_pretrained(model_name)
                # save for future use
                tokenizer.save_pretrained(F"pretrained/{safe_model_name}_tokenizer")
                _model.save_pretrained(F"pretrained/{safe_model_name}_model")
        case "allenai/specter2_aug2023refresh_base":
            try:
                tokenizer = AutoTokenizer.from_pretrained(F"pretrained/{safe_model_name}_tokenizer")
                _model = AutoAdapterModel.from_pretrained(F"pretrained/{safe_model_name}_tokenizer_model")
                _model.load_adapter("allenai/specter2_aug2023refresh_classification", source="hf", load_as="specter2_classification", set_active=True)
            except:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                _model = AutoAdapterModel.from_pretrained(model_name)
                _model.load_adapter("allenai/specter2_aug2023refresh_classification", source="hf", load_as="specter2_classification", set_active=True)
                # save for future use
                tokenizer.save_pretrained(F"pretrained/{safe_model_name}_tokenizer")
                _model.save_pretrained(F"pretrained/{safe_model_name}_model") # TODO: Test if adapter gets saved also.

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # use whatever pad token is most applicable to the model

    if hasattr(_model.config,"pad_token_id"):
        _model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    _model.resize_token_embeddings(len(tokenizer)) # Cursed... see https://github.com/huggingface/transformers/issues/487
    _model.to(device)

#--------------------------#
#   tokenization methods   #
#--------------------------#
def to_tokens(x:pd.DataFrame, do_title=True, do_abstract=True, batch_size:int=32) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Tokenizes the titles and abstracts from a DataFrame into a list of batched titles and abstracts.
    Returns: titles, abstracts (both tokenized & batched)"""
    title_token_batches = []
    abstract_token_batches = []

    for i in tqdm(range(0, len(x), batch_size)):
        batch = x.iloc[i:min(i+batch_size,len(x))]
        tit_batch = batch.iloc[:, 0].tolist()
        abs_batch = batch.iloc[:, 1].tolist()
        if do_title:
            tit_token_batch = tokenizer(tit_batch, return_tensors="pt", padding=True)
            title_token_batches.append(tit_token_batch)
        if do_abstract:
            abs_token_batch = tokenizer(abs_batch, return_tensors="pt", padding=True)
            abstract_token_batches.append(abs_token_batch)
        
    return title_token_batches, abstract_token_batches

def to_prompt_tokens(x:pd.DataFrame, prompter:Callable[[str,str],str], truncate=None, batch_size:int= 32) -> List[torch.Tensor]:
    """Combines the title and abstract via a to_prompt lambda and tokenizes them into batches.
    Returns: prompts (tokenized and batched)"""
    prompt_token_batches = []

    for i in tqdm(range(0, len(x), batch_size), desc="Tokenizing prompts"):
        batch = x.iloc[i:min(i+batch_size,len(x))]
        tit_batch = batch.iloc[:, 0].tolist()
        abs_batch = batch.iloc[:, 1].tolist()
        prompt_batch = []
        for title, abstract in zip(tit_batch, abs_batch):
            prompt_batch.append(prompter(title,abstract))
        prompt_tokens_batch = tokenizer(prompt_batch, return_tensors="pt", padding='max_length', max_length=truncate, truncation=(True if truncate else False))
        prompt_token_batches.append(prompt_tokens_batch)

    return prompt_token_batches

#-----------------------------#
#   embedding architectures   #
#-----------------------------#
def to_decoder_last_layer_mean_embeddings(x:pd.DataFrame, 
                                      prompter:Callable[[str,str],str], 
                                      batch_size:int=32, 
                                      name:Optional[str]=None) -> List[torch.Tensor]:
    """Create embedding from mean of last layer of a decoder."""
    embeddings = []

    prompt_token_batches = to_prompt_tokens(x, prompter, batch_size)
    for prompt_batch in tqdm(prompt_token_batches, desc="Embedding prompts"):
        prompt_batch.to(device)
        with torch.no_grad():
            outputs = _model(**prompt_batch)
        embeddings.append(outputs.last_hidden_state.mean(dim=1))
        del prompt_batch, outputs # should force-deallocate when we call collect/empty_cache
        torch.cuda.empty_cache()
        gc.collect()
    if name:
        torch.save(embeddings, name.replace("/","_"))
    return embeddings

def to_raw_embeddings(x:pd.DataFrame, 
                     prompter:Callable[[str,str],str], 
                     batch_size:int=32,
                     truncate=512, 
                     name:Optional[str]=None) -> List[torch.Tensor]:
    """"""
    embeddings = []
    prompt_token_batches = to_prompt_tokens(x, prompter=prompter, batch_size=batch_size, truncate=truncate)

    for prompt_batch in tqdm(prompt_token_batches):
        prompt_batch.to(device)
        with torch.no_grad():
            outputs = _model(**prompt_batch)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu()) # take the first token in the batch as the embedding
        del prompt_batch, outputs # should force-deallocate when we call collect/empty_cache
        torch.cuda.empty_cache()
        gc.collect()
    if name:
        torch.save(embeddings, name.replace("/","_"))
    return embeddings

#----------------------------#
#   datasets & dataloaders   #
#----------------------------#
def split_dataloader(dataset:Dataset, batch_size=32, split:float=0.8) -> tuple[Dataset, Dataset]:
    train_split = int(len(dataset)*split)
    val_split = len(dataset) - train_split
    train_dataset, val_dataset = random_split(dataset,[train_split, val_split])
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

class EmbeddingDataset(Dataset):
    def __init__(self, x:pd.DataFrame, y:pd.DataFrame, 
                 embedder:Callable[[pd.DataFrame,Callable[[str,str],str],Optional[int],Optional[str]],List[torch.Tensor]], 
                 prompter:Callable[[str,str],str]=(lambda t,a: t + (tokenizer.sep_token if tokenizer.sep_token else "\n") + a),
                 partial_credit:Optional[float]=None, 
                 name=None,
                 batch_size=32):
        file = F".{model_name}_{name}_embeddings.pt".replace("/","_")
        # embed if embeddings have not been computed before
        if os.path.isfile(file):
            #print("Found pre-calculated embeddings, loading...")
            x_batch_list = torch.load(file)
        else:
            #print("No pre-computed embeddings found, calculating...")
            x_batch_list = embedder(x, prompter=prompter, batch_size=batch_size, name=file)
        #print("Embeddings received, creating dataset...")
        # de-batch x
        self.x = []
        for x_batch in x_batch_list:
            for x in x_batch:
                self.x.append(x)

        label_indices = y['label'].to_numpy()
        _y = torch.zeros((len(y), 151), device=device)
        _y[torch.arange(len(y)), label_indices] = 1.0
        if partial_credit:
            secondary_indices = y['all_subfields_numeric'].apply(lambda x: [int(lbl) for lbl in str(x).split() if lbl.isdigit()])
            for i, secondary_labels in enumerate(secondary_indices):
                primary_label = label_indices[i]
                secondary_labels = [lbl for lbl in secondary_labels if lbl != primary_label]
                if secondary_labels:
                    for j, seclbl in enumerate(secondary_labels):
                        # constant weighting
                        #_y[i, secondary_labels] = partial_credit/len(secondary_labels)
                        #_y[i, primary_label] = 1 - partial_credit
                        # exponential weighting (assumes order of sub-subcategories)
                        _y[i, seclbl] = partial_credit/(2**(j+1))
                        _y[i, primary_label] = 1 - partial_credit/(2**len(secondary_labels))
        self.y = _y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class JointEmbeddingDataset(Dataset):
    def __init__(self, embeddings1:str, 
                 embeddings2:str, 
                 embeddings3:str,
                 y:pd.DataFrame, 
                 partial_credit:Optional[float]=None, 
                 name=None,
                 batch_size=32):
        x1_batch_list = torch.load(embeddings1) # (total_batches x 512) x 768 
        x2_batch_list = torch.load(embeddings2) 
        x3_batch_list = torch.load(embeddings3) # (total_batches x 32) x 1600
        
        x1_flat_list = [x1 for x1_batch in x1_batch_list for x1 in x1_batch.to(device)]
        x2_flat_list = [x2 for x2_batch in x2_batch_list for x2 in x2_batch.to(device)]
        x3_flat_list = [x3 for x3_batch in x3_batch_list for x3 in x3_batch.to(device)]
        #print("Embeddings received, creating dataset...")
        # de-batch x
        self.x = []
        for x1,x2,x3 in zip(x1_flat_list, x2_flat_list, x3_flat_list):
            self.x.append(torch.cat((x1,x2,x3)))

        label_indices = y['label'].to_numpy()
        _y = torch.zeros((len(y), 151), device=device)
        _y[torch.arange(len(y)), label_indices] = 1.0
        if partial_credit:
            secondary_indices = y['all_subfields_numeric'].apply(lambda x: [int(lbl) for lbl in str(x).split() if lbl.isdigit()])
            for i, secondary_labels in enumerate(secondary_indices):
                primary_label = label_indices[i]
                secondary_labels = [lbl for lbl in secondary_labels if lbl != primary_label]
                if secondary_labels:
                    for j, seclbl in enumerate(secondary_labels):
                        # constant weighting
                        #_y[i, secondary_labels] = partial_credit/len(secondary_labels)
                        #_y[i, primary_label] = 1 - partial_credit
                        # exponential weighting (assumes order of sub-subcategories)
                        _y[i, seclbl] = partial_credit/(2**(j+1))
                        _y[i, primary_label] = 1 - partial_credit/(2**len(secondary_labels))
        self.y = _y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class EvaluationDataset(Dataset):
    def __init__(self, x:pd.DataFrame, 
                 embedder:Callable[[pd.DataFrame,Callable[[str,str],str],Optional[int],Optional[str]],List[torch.Tensor]], 
                 prompter:Callable[[str,str],str]=(lambda t,a: t + (tokenizer.sep_token if tokenizer.sep_token else "\n") + a),
                 name=None,
                 batch_size=32):
        file = F".{model_name}_{name}_embeddings.pt".replace("/","_")
        
        # embed if embeddings have not been computed before
        if os.path.isfile(file):
            #print("Found pre-calculated embeddings, loading...")
            x_batch_list = torch.load(file)
        else:
            #print("No pre-computed embeddings found, calculating...")
            x_batch_list = embedder(x, prompter=prompter, batch_size=batch_size, name=file)
        #print("Embeddings received, creating dataset...")
        # de-batch x
        self.x = []
        for x_batch in x_batch_list:
            for x in x_batch:
                self.x.append(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]
    
class JointEvaluationDataset(Dataset):
    def __init__(self, 
                 embeddings1,
                 embeddings2,
                 embeddings3,
                 name=None,
                 batch_size=32):
        
        x1_batch_list = torch.load(embeddings1) # (total_batches x 512) x 768 
        x2_batch_list = torch.load(embeddings2) 
        x3_batch_list = torch.load(embeddings3) # (total_batches x 32) x 1600
        
        x1_flat_list = [x1 for x1_batch in x1_batch_list for x1 in x1_batch.to(device)]
        x2_flat_list = [x2 for x2_batch in x2_batch_list for x2 in x2_batch.to(device)]
        x3_flat_list = [x3 for x3_batch in x3_batch_list for x3 in x3_batch.to(device)]
        #print("Embeddings received, creating dataset...")
        # de-batch x
        self.x = []
        for x1,x2,x3 in zip(x1_flat_list, x2_flat_list, x3_flat_list):
            self.x.append(torch.cat((x1,x2,x3)))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]
#------------#
#   models   #
#------------#

class SimpleEmbeddingClassifier(nn.Module):
    def __init__(self, din, dhidden, dout, p):
        super().__init__()
        self.fc1 = nn.Linear(din, dhidden, device=device)
        self.d1 = nn.Dropout(p)
        self.fc2 = nn.Linear(dhidden, dout, device=device)

    def forward(self, x):
        x = self.d1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def sequence_classification_zero_shot(x:pd.DataFrame,
                                      premise:Callable[[str,str],str]=(lambda title,abstract:F"Title: {title}\nAbstract: {abstract}"),
                                      hypothesis:Callable[[str,str],str]=(lambda label:F"This paper is in the arXiv subcategory {label}.")):
    """Use a sequence (zero-shot) classification model to predict subclasses."""
    pass

# NOTE: This paper [https://arxiv.org/pdf/2405.04136] inspired both the use of SPECTER2 and fine-tuning over using raw features.

class ScientificSequenceClassifier(nn.Module):
    def __init__(self, din, dhidden, dout):
        super().__init__()

    def forward(self, x):
        return x


#--------------------------------#
#   training, validation loops   #
#--------------------------------#
def train(model, dataloader, optimizer, criterion, epoch, sumwriter):
    model.train()
    total_loss = 0
    for x,y in dataloader:
        x,y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(dataloader)
    #print(F"Loss: {avg_loss}")
    sumwriter.add_scalar('Loss (Train)', avg_loss, epoch)    
    return avg_loss

def validate(model, dataloader, criterion, epoch, sumwriter):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  
        for x, y in (dataloader):
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            correct_predictions += (y_pred.argmax(dim=1) == (y if (len(y.size()) == 1) else y.argmax(dim=1))).sum().item()
            total_predictions += y.size(0)
    avg_loss = total_loss/len(dataloader)
    accuracy = correct_predictions / total_predictions
    #print(f"Validation Loss:{avg_loss} Accuracy:{accuracy}")
    sumwriter.add_scalar('Loss (Validation)', avg_loss, epoch)
    sumwriter.add_scalar('Accuracy (Validation)', accuracy, epoch)
    return avg_loss, accuracy

#-----------------------------#
#   output test predictions   #
#-----------------------------#

def evaluate(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():  
        for x in (dataloader):
            x = x.to(device)
            output = model(x)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.flatten())
    return predictions

def save_predictions(ids, preds, filename="kaggle_submission.csv"):
  output_dict = {'id': ids, 'label': preds}
  #breakpoint()
  output = pd.DataFrame.from_dict(output_dict)
  output.to_csv(filename, index=False)
  return
#---------------------------------#
#   special notes and resources   #
#---------------------------------#
# https://arxiv.org/category_taxonomy