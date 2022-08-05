import json, time
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
tokenizer_name = "mbert"

batch_size = 16

import torch
import torch.nn as nn

if torch.cuda.is_available():       
  device = torch.device("cuda")
  print("Using GPU.")
else:
  print("No GPU available, using the CPU instead.")
  device = torch.device("cpu")

# Data Processing
def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(input_data, input_dir):
  files_present,files_absent = 0,0
  inputs1, inputs2 = list(), list()
  masks1, masks2 = list(), list()
  targets = list()
  for i in range(len(input_data)):
    row = input_data.iloc[i]
    pair = row['pair_id']
    
    y_target = row['Overall']
    # y_scaler = StandardScaler()
    # y_target = y_scaler.transform(y_target.reshape(-1, 1))

    f1, f2 = pair.split("_")
    folder1 = f1[-2:]
    folder2 = f2[-2:]
    try:
      with open(input_dir+"/"+folder1+"/"+f1+".json","r") as f:
        d1 = json.load(f)
      with open(input_dir+"/"+folder2+"/"+f2+".json","r") as f:
        d2 = json.load(f)
    except Exception as E:
      files_absent = files_absent + 1
      continue
    
    files_present = files_present+1

    s1 = text_preprocessing(d1['text'])
    s2 = text_preprocessing(d2['text'])
    encoded_input1 = tokenizer.encode_plus(s1, return_tensors='pt',return_attention_mask=True, pad_to_max_length=True,max_length=256)
    encoded_input2 = tokenizer.encode_plus(s2, return_tensors='pt',return_attention_mask=True, pad_to_max_length=True,max_length=256)

    input_ids1 = encoded_input1['input_ids']
    input_ids2 = encoded_input2['input_ids']
    attention_masks1 = encoded_input1['attention_mask']
    attention_masks2 = encoded_input2['attention_mask']
    
    inputs1.append(input_ids1)
    inputs2.append(input_ids2)
    masks1.append(attention_masks1)
    masks2.append(attention_masks2)
    targets.append(y_target)

  print(f"Files present: {files_present} and Files absent: {files_absent}. % missing = {(files_absent*100)/(files_present+files_absent)}%")
  return inputs1, inputs2, masks1, masks2, targets  

def create_dataloaders(inputs1, inputs2, masks1, masks2, targets, batch_size):
    input_tensor1 = torch.tensor([t.numpy() for t in inputs1])
    input_tensor2 = torch.tensor([t.numpy() for t in inputs2])
    mask_tensor1 = torch.tensor([t.numpy() for t in masks1])
    mask_tensor2 = torch.tensor([t.numpy() for t in masks2])
    targets_tensor = torch.tensor(targets)
    dataset = TensorDataset(input_tensor1, input_tensor2, mask_tensor1, mask_tensor2, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    return dataloader

############################################################# Modeling Baselines   #################################################################

#Model Architecure

#Baseline 1: mBERT
class mBERTRegressor(nn.Module):
    
    def __init__(self, drop_rate=0.2, freeze_camembert=False):        
        super(mBERTRegressor, self).__init__()
        D_in, D_out = 1536, 1
        self.mbert = AutoModel.from_pretrained("bert-base-multilingual-cased")   
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)) 
           
    def forward(self, input_ids1, input_ids2, attention_masks1, attention_masks2):
      outputs1 = self.mbert(input_ids1, attention_masks1)
      outputs2 = self.mbert(input_ids2, attention_masks2)
      #outputs = torch.dot(outputs1[1],outputs2[1])
      #outputs = outputs1[1]+outputs2[1]
      outputs = torch.cat((outputs1[1], outputs2[1]), 1)
    
      # last_hidden_state_cls = outputs[0][:, 0, :]
      last_hidden_state_cls = outputs
      final_outputs = self.regressor(last_hidden_state_cls)
      return final_outputs

      
#Baseline 2: XLM
class XLMRegressor(nn.Module):
    
    def __init__(self, drop_rate=0.2, freeze_camembert=False):        
        super(XLMRegressor, self).__init__()
        D_in, D_out = 1536, 1
        self.xlm = AutoModel.from_pretrained("xlm-roberta-base")
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)) 
           
    def forward(self, input_ids1, input_ids2, attention_masks1, attention_masks2):
      outputs1 = self.xlm(input_ids1, attention_masks1)
      outputs2 = self.xlm(input_ids2, attention_masks2)
      #outputs = torch.dot(outputs1[1],outputs2[1])
      #outputs = outputs1[1]+outputs2[1]
      outputs = torch.cat((outputs1[1], outputs2[1]), 1)
      # last_hidden_state_cls = outputs[0][:, 0, :]
      last_hidden_state_cls = outputs
      final_outputs = self.regressor(last_hidden_state_cls)
      return final_outputs






freeze = False #set to True for 3rd baseline

epochs = 10
total_steps = len(train_dataloader) * epochs
if freeze:
  optimizer = AdamW(model.regressor.parameters(), lr=5e-5, eps=1e-8) 
else:
  optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)    
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0, num_training_steps=total_steps)
loss_function = nn.MSELoss()


def train(model, optimizer, scheduler, loss_function, epochs, train_dataloader, device, clip_value=2):
    for epoch_i in range(epochs):
        print(f"Epoch {epoch_i}")
        print("------------------------------------------------------------")
        best_loss = 1e10

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        
        model.train()
        for step, batch in enumerate(train_dataloader): 
            batch_counts +=1
            
            batch_inputs1, batch_inputs2, batch_masks1, batch_masks2, batch_targets = tuple(b.to(device) for b in batch)
            batch_inputs1 = batch_inputs1.squeeze(1)
            batch_inputs2 = batch_inputs2.squeeze(1)
            batch_masks1 = batch_masks1.squeeze(1)
            batch_masks2 = batch_masks2.squeeze(1)
            
            model.zero_grad()
            
            outputs = model(batch_inputs1, batch_inputs2, batch_masks1, batch_masks2)           
            
            loss = loss_function(outputs.squeeze(), batch_targets.float().squeeze())
            batch_loss += loss.item()
            total_loss += loss.item()
            
            loss.backward()
            
            if freeze:
              clip_grad_norm(model.regressor.parameters(), clip_value)
            else:
              clip_grad_norm(model.parameters(), clip_value)
            
            optimizer.step()
            scheduler.step()
            
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
              
#             with torch.no_grad():
#                 del outputs
#                 torch.cuda.empty_cache()
        
#             show_gpu(f'{epoch_i}: GPU memory usage after training model:')
            del loss, outputs
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
#             show_gpu(f'{epoch_i}: GPU memory usage after clearing cache:')
        
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f}")
        print();

    return model



######################################################################### Evaluation  #############################################################33

def pearson(X,Y):
    print(X,Y)
    xm = torch.mean(X.float())
    ym = torch.mean(Y.float())
    print(xm, ym)
    num = 0.0
    deno1 = 0.0
    deno2 = 0.0
    deno = 0.0
    n = X.shape[0]
    for i in range(n):
        num += (X[i]-xm)*(Y[i]-ym)
        deno1 += (X[i]-xm)*(X[i]-xm)
        deno2 += (Y[i]-ym)*(Y[i]-ym)
    deno = deno1*deno2
    deno = torch.sqrt(deno)
    return (num/deno).item()

def evaluate(model, loss_function, test_dataloader, device):
    model.eval()
    test_loss, test_pearson, test_mse = [], [], []
    for batch in test_dataloader:
        batch_inputs1, batch_inputs2, batch_masks1, batch_masks2, batch_targets = tuple(b.to(device) for b in batch)
        batch_inputs1 = batch_inputs1.squeeze(1)
        batch_inputs2 = batch_inputs2.squeeze(1)
        batch_masks1 = batch_masks1.squeeze(1)
        batch_masks2 = batch_masks2.squeeze(1)
        with torch.no_grad():
            outputs = model(batch_inputs1, batch_inputs2, batch_masks1, batch_masks2)   

        loss = loss_function(outputs, batch_targets)
        test_loss.append(loss.item())
        preds = outputs.cpu()
        targs = batch_targets.cpu()
        # print(preds, targs)
        pearson_score = pearson(preds, targs)
        MSE = np.square(np.subtract(preds, targs)).mean()
        test_pearson.append(pearson_score)
        test_mse.append(MSE)
    return test_loss, test_pearson, test_mse


#Loading Train and Test Dataset Files
train_data = pd.read_csv("train_v2.csv")
test_data = pd.read_csv("eval_with_result.csv")

#Train Set
train_inputs1, train_inputs2, train_masks1, train_masks2, train_targets = load_data(train_data, "articles")
train_dataloader = create_dataloaders(train_inputs1, train_inputs2, train_masks1, train_masks2, train_targets, batch_size)

#Test Set
test_inputs1, test_inputs2, test_masks1, test_masks2, test_targets = load_data(test_data, "eval_data")
test_dataloader = create_dataloaders(test_inputs1, test_inputs2, test_masks1, test_masks2, test_targets, batch_size)

model = mBERTRegressor(drop_rate=0.2).to(device)

# %%time
model = train(model, optimizer, scheduler, loss_function, epochs,train_dataloader, device, clip_value=2)


test_loss, test_pearson, test_mse = evaluate(model, loss_function, test_dataloader, device)

print(f"Test Pearson Score is {np.sum(test_pearson)/len(test_pearson)}")
print(f"Test MSE Score is {np.sum(test_mse)/len(test_mse)}")

