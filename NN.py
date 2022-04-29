
import gensim.downloader
from lightgbm import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

def remove_punct(text):
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“”’‘—'
    replace_with_space = '/—_[\\]<=>:;'
    for ch in punc:
        if ch in text:
            if ch not in replace_with_space:
                text = text.replace(ch,'')
            else:
                text = text.replace(ch,' ')
    return text

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
    
    return total_loss/len(train_loader.dataset)    

def test_model(model,criterion,loader):
    model.eval()
    total_loss = 0
    for data in loader:
        x = data[0]
        label = data[1]
        pred = model(x)
        loss = criterion(pred,label)
        total_loss += loss.item()
    
    return total_loss/len(test_loader.dataset)

class SuicideNet(nn.Module):

    def __init__(self, input_words=500):
        super().__init__()
        self.fc1 = nn.Linear(input_words*25,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

# corpus_valuable = pd.read_csv('valuable.csv')
# corpus_valuable.dropna(inplace=True)
# #separate classes and text
# corpus = {'text':[],'class':[]}
# for i in range(corpus_valuable.size):
#     single_data = corpus_valuable.values[i][0].split(',')
#     corpus['text'].append(",".join(single_data[:-1]))
#     corpus['class'].append(single_data[-1])

# my_corpus = pd.DataFrame(corpus,columns = ['text','class'])

corpus_reddit = pd.read_csv('reddit_corpus_agree.csv')
corpus_reddit['label'] = 0
corpus_reddit['label'].loc[(corpus_reddit['cls']=='Risk')] = 1 
del corpus_reddit['cls']

#remove punctuations
for i in range(len(corpus_reddit['text'])):
    text = corpus_reddit['text'].iloc[i]
    text_clean = remove_punct(text)
    corpus_reddit['text'].iloc[i] = text_clean

#load pretrained glove model
glove_vectors = gensim.downloader.load('glove-twitter-25')

#create embeddings for data
split_corpus = corpus_reddit.text.str.split()
embeddings = []
for i in range(len(split_corpus)):
    text = split_corpus[i]
    sentence = []
    for word in text:
        if (not word.isnumeric()):
            try:
                word_embed = glove_vectors.word_vec(word.lower())
            except:
                word_embed = np.zeros(25)
            sentence.append(torch.Tensor(word_embed))
    #make all inputs same size as they will be input of NNs.       
    if len(sentence)>=500:
        embeddings.append(np.stack(sentence[:500],axis=0).flatten())
    else:
        for pad in range(500-len(sentence)):
            sentence.append(np.zeros(25,dtype='float32'))
        embeddings.append(np.stack(sentence,axis=0).flatten())

corpus_reddit['embeddings'] = embeddings
#create a class balanced data split
X_train, X_val, y_train, y_val = train_test_split(corpus_reddit['embeddings'], corpus_reddit['label'], test_size=0.1, stratify=corpus_reddit['label'])
train_dataset = data_utils.TensorDataset(torch.tensor(np.stack(X_train.values)) , torch.tensor(y_train.values,dtype=torch.float32).view(-1,1))
test_dataset = data_utils.TensorDataset(torch.tensor(np.stack(X_val.values)) , torch.tensor(y_val.values,dtype=torch.float32).view(-1,1))

train_loader = data_utils.DataLoader(train_dataset,batch_size=10,shuffle=True)
test_loader = data_utils.DataLoader(test_dataset,batch_size=1)

my_NN = SuicideNet()
optimizer = torch.optim.Adam(my_NN.parameters(),lr=0.01)
criterion = nn.BCELoss()

for epoch in range(10):
    loss = train_model(my_NN,optimizer,criterion,train_loader)
    print(f"Epoch {epoch+1}, train_loss: {loss}")
    test_loss = test_model(my_NN,criterion,test_loader)
    print(f"test_loss: {test_loss}")

