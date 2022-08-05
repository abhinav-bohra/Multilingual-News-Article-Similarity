import json, time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr

if torch.cuda.is_available():       
  device = torch.device("cuda")
  print("Using GPU.")
else:
  print("No GPU available, using the CPU instead.")
  device = torch.device("cpu")


def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(input_data, input_dir, max_len=256):
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
    s1 = text_preprocessing(d1['title']+"[SEP]"+d1['text'])
    s2 = text_preprocessing(d2['title']+"[SEP]"+d2['text'])
    
    files_present = files_present+1
    encoded_input1 = tokenizer.encode_plus(s1, return_tensors='pt',return_attention_mask=True, pad_to_max_length=True,max_length=max_len)
    encoded_input2 = tokenizer.encode_plus(s2, return_tensors='pt',return_attention_mask=True, pad_to_max_length=True,max_length=max_len)

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

#Model Architecure
class STRegressor(nn.Module):
    
    def __init__(self, drop_rate=0.2):        
        super(STRegressor, self).__init__()
        D_in, D_out = 768*2, 1
        self.ST = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")   
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)) 
    def mean_pool(self, token_embeds, attention_mask):
        # reshape attention_mask to cover 768-dimension embeddings
        in_mask = attention_mask.unsqueeze(-1).expand(
            token_embeds.size()
        ).float()
        # perform mean-pooling but exclude padding tokens (specified by in_mask)
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
            in_mask.sum(1), min=1e-9
        )
        return pool
    def forward(self, input_ids1, input_ids2, attention_masks1, attention_masks2):
        outputs1 = self.mean_pool(self.ST(input_ids1, attention_masks1)[0], attention_masks1)
        outputs2 = self.mean_pool(self.ST(input_ids2, attention_masks2)[0], attention_masks2)
        #outputs = torch.dot(outputs1,outputs2)
        #outputs = outputs1[1]+outputs2[1]
        #outputs = torch.cat((outputs1, outputs2), 1)
        # last_hidden_state_cls = outputs[0][:, 0, :]
        #last_hidden_state_cls = outputs
        #final_outputs = self.regressor(last_hidden_state_cls)
        return outputs1, outputs2

def loss_fn(labels,  o1=None, o2=None):
    labels = 1-(labels-1)/3
    cosine_similarity = nn.CosineSimilarity()(o1, o2)
    cs_loss = nn.MSELoss()(cosine_similarity, labels) 
    loss = cs_loss
    return loss

def renormalise_similarity_score(scores):
    # reverse the normalisation I did for SBERT finetuning
    scores = np.array(scores)
    renormalised_scores = (3 - (scores * 3)) + 1
    return renormalised_scores

def pearson(X,Y):
    #print(X,Y)
    xm = torch.mean(X.float())
    ym = torch.mean(Y.float())
    #print(xm, ym)
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
    predicted=[]
    targets=[]
    cosine_sims=[]
    for batch in tqdm(test_dataloader):
        batch_inputs1, batch_inputs2, batch_masks1, batch_masks2, batch_targets = tuple(b.to(device) for b in batch)
        batch_inputs1 = batch_inputs1.squeeze(1)
        batch_inputs2 = batch_inputs2.squeeze(1)
        batch_masks1 = batch_masks1.squeeze(1)
        batch_masks2 = batch_masks2.squeeze(1)
        with torch.no_grad():
            o1, o2 = model(batch_inputs1, batch_inputs2, batch_masks1, batch_masks2)   

        #loss = loss_function(outputs, batch_targets, None, None)
        #test_loss.append(loss.item())
        #print(outputs.shape, batch_targets.shape)
        #preds = outputs.cpu().squeeze(1).numpy()
        targs = batch_targets.cpu().numpy()
        #predicted +=list(preds)
        targets += list(targs)
        cosine_sims += list(renormalise_similarity_score(nn.CosineSimilarity()(o1, o2).cpu().numpy()))
    #print(len(predicted))
    #pearson_score1 = pearson(torch.tensor(renormalise_similarity_score(predicted)), torch.tensor(targets))
    pearson_score2 = pearson(torch.tensor(cosine_sims), torch.tensor(targets))
    return pearson_score2

def train(model, optimizer, scheduler, loss_fn, epochs, train_dataloader, device, clip_value=2):
    best_so_far_pearson=-1.0
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
            
            o1, o2 = model(batch_inputs1, batch_inputs2, batch_masks1, batch_masks2)           
            
            loss = loss_fn(batch_targets.float().squeeze(), o1, o2)
            batch_loss += loss.item()
            total_loss += loss.item()
            
            loss.backward()
            
            if freeze:
              clip_grad_norm(model.regressor.parameters(), clip_value)
            else:
              clip_grad_norm(model.parameters(), clip_value)
            
            optimizer.step()
            scheduler.step()
            
            if (step % 50 == 0 and step != 0) or (step == len(train_dataloader) - 1):
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
            del loss
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
#             show_gpu(f'{epoch_i}: GPU memory usage after clearing cache:')
        
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f}")
        val_pearson_cos = evaluate(model, loss_fn, val_dataloader, device)
        print("Validation: Pearson Corr Cosine-",val_pearson_cos)
        if val_pearson_cos>best_so_far_pearson:
          best_so_far_pearson = val_pearson_cos
          torch.save(model, "ST.pt")
        print();

    return model

#Loading Train and Test Dataset Files
train_data = pd.read_csv("./train.csv")
val_data = pd.read_csv("./val.csv")
test_data = pd.read_csv("eval_with_result.csv")

#tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
#tokenizer_name = "mbert"

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
tokenizer_name = "SBERT"

#tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
#tokenizer_name = "xlm"

batch_size = 24

#Train Set
train_inputs1, train_inputs2, train_masks1, train_masks2, train_targets = load_data(train_data, "articles", max_len=128)
train_dataloader = create_dataloaders(train_inputs1, train_inputs2, train_masks1, train_masks2, train_targets, batch_size)

#Train Set
val_inputs1, val_inputs2, val_masks1, val_masks2, val_targets = load_data(val_data, "articles", max_len=128)
val_dataloader = create_dataloaders(val_inputs1, val_inputs2, val_masks1, val_masks2, val_targets, batch_size)

#Test Set
test_inputs1, test_inputs2, test_masks1, test_masks2, test_targets = load_data(test_data, "eval_data", max_len=128)
test_dataloader = create_dataloaders(test_inputs1, test_inputs2, test_masks1, test_masks2, test_targets, batch_size)

freeze = False
epochs = 10
total_steps = len(train_dataloader) * epochs
if freeze:
  optimizer = AdamW(model.regressor.parameters(), lr=1e-5, eps=1e-8) 
else:
  optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)    
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0, num_training_steps=total_steps)

model = STRegressor(drop_rate=0.2).to(device)

print("Initial Pearson Corr Train: ",evaluate(model, loss_fn, train_dataloader, device))
print("Initial Pearson Corr Val: ", evaluate(model, loss_fn, val_dataloader, device))
print("Initial Pearson Corr Test: ", evaluate(model, loss_fn, test_dataloader, device))

# %%time
model = train(model, optimizer, scheduler, loss_fn, epochs,train_dataloader, device, clip_value=2)

### Evaluation
model1 = torch.load("ST.pt")
print("Final Pearson Corr Train: ",evaluate(model1, loss_fn, train_dataloader, device))
print("Final Pearson Corr Val: ", evaluate(model1, loss_fn, val_dataloader, device))
print("Final Pearson Corr Test: ", evaluate(model1, loss_fn, test_dataloader, device))

