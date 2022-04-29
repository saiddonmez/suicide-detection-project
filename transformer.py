import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertConfig
import torch.utils.data as data_utils
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig
# def pad_sequence(sequence,max_length):
#     if len(sequence)<max_length:
#         sequence += (max_length-len(sequence))*[0]
#     return sequence

def embeddings_from_dataset(X, tokenizer, bert_model):

    X = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in X] # Appending [CLS] and [SEP] tokens - this probably can be done in a cleaner way
    #X_test = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in X_test] # Appending [CLS] and [SEP] tokens - this probably can be done in a cleaner way
    X = [text[:512] if len(text)>512 else text for text in X]
    #X_test = [text[:512] if len(text)>512 else text for text in X_test]
    X_tokens = [tokenizer.convert_tokens_to_ids(sent) for sent in X]
    #X_tokens = [pad_sequence(sequence) if len(sequence)<512 else sequence for sequence in X_tokens]

    #X_test_tokens = [tokenizer.convert_tokens_to_ids(sent) for sent in X_test]

    train_embeddings = []
    #test_embeddings = []

    #results = torch.zeros((len(X_train_tokens), bert_model.config.hidden_size)).long()
    with torch.no_grad():
        for stidx in range(len(X_tokens)):
            tokens = X_tokens[stidx]
            tokens_t = torch.LongTensor(tokens)#.to(device)
            segment_t = torch.LongTensor([1] * len(tokens))#.to(device)
            outputs = bert_model(tokens_t.unsqueeze(0),segment_t.unsqueeze(0))
            embeddings = outputs[0][0][0] #This only takes CLS embedding
            train_embeddings.append(embeddings.cpu())
            #results[stidx] = embeddings.cpu()
        
        # for stidx in range(len(X_test)):
        #     tokens = X_test_tokens[stidx]
        #     tokens_t = torch.LongTensor(tokens)#.to(device)
        #     segment_t = torch.LongTensor([1] * len(tokens))#.to(device)
        #     outputs = bert_model(tokens_t.unsqueeze(0),segment_t.unsqueeze(0))
        #     embeddings = outputs[0][0][0] #This only takes CLS embedding
        #     test_embeddings.append(embeddings.cpu())
    return torch.stack(train_embeddings)

def train_model(model,optimizer,criterion,loader):
    model.train()
    total_loss = 0
    for data in loader:
        x = data[0]
        label = data[1]
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred,label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss/len(loader.dataset)    

def test_model(model,criterion,loader):
    model.eval()
    total_loss = 0
    for data in loader:
        x = data[0]
        label = data[1]
        pred = model(x)
        loss = criterion(pred,label)
        total_loss += loss.item()
    
    return total_loss/len(loader.dataset)
class BertClassifier(nn.Module):
    def __init__(self, input_size = 768):
        super().__init__()
        self.fc1 = nn.Linear(input_size,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x



corpus_reddit = pd.read_csv('reddit_corpus_agree.csv')
corpus_reddit['label'] = 0
corpus_reddit['label'].loc[(corpus_reddit['cls']=='Risk')] = 1 
del corpus_reddit['cls']

batch_size = 1
texts, labels =  corpus_reddit['text'], corpus_reddit['label']
#texts = [" ".join(text.split()[:512]) if len(text.split())>512 else text for text in texts]


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, stratify=corpus_reddit['label'])
model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = BertConfig.from_pretrained(model_name, output_hidden_states=True)    
bert_model = BertModel.from_pretrained(model_name, config=config)
bert_model = bert_model#.to(device)
bert_model.eval()

train_embeddings = embeddings_from_dataset(X_train, tokenizer, bert_model)
test_embeddings = embeddings_from_dataset(X_test, tokenizer, bert_model)

train_dataset = data_utils.TensorDataset(train_embeddings , torch.FloatTensor(y_train.values).view(-1,1))
test_dataset = data_utils.TensorDataset(test_embeddings , torch.FloatTensor(y_test.values).view(-1,1))

train_loader = data_utils.DataLoader(train_dataset,batch_size=10,shuffle=True)
test_loader = data_utils.DataLoader(test_dataset,batch_size=1)

my_NN = BertClassifier()
optimizer = torch.optim.Adam(my_NN.parameters(),lr=0.01)
criterion = nn.BCELoss()

for epoch in range(10):
    loss = train_model(my_NN,optimizer,criterion,train_loader)
    print(f"Epoch {epoch+1}, train_loss: {loss}")
    test_loss = test_model(my_NN,criterion,test_loader)
    print(f"test_loss: {test_loss}")