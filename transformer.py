import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertConfig
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig


corpus_reddit = pd.read_csv('reddit_corpus_agree.csv')
corpus_reddit['label'] = 0
corpus_reddit['label'].loc[(corpus_reddit['cls']=='Risk')] = 1 
del corpus_reddit['cls']

batch_size = 1
texts, labels =  corpus_reddit['text'], corpus_reddit['label']
#texts = [" ".join(text.split()[:512]) if len(text.split())>512 else text for text in texts]


X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, stratify=corpus_reddit['label'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in X_train] # Appending [CLS] and [SEP] tokens - this probably can be done in a cleaner way
X_test = [tokenizer.tokenize('[CLS] ' + sent + ' [SEP]') for sent in X_test] # Appending [CLS] and [SEP] tokens - this probably can be done in a cleaner way
X_train = [" ".join(text.split()[:512]) if len(text.split())>512 else text for text in X_train]
X_test = [" ".join(text.split()[:512]) if len(text.split())>512 else text for text in X_test]




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = BertConfig.from_pretrained( 'bert-base-uncased', output_hidden_states=True)    
bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)
bert_model = bert_model#.to(device)
bert_model.eval()
X_train_tokens = [tokenizer.convert_tokens_to_ids(sent) for sent in X_train]
results = []
#results = torch.zeros((len(X_train_tokens), bert_model.config.hidden_size)).long()
with torch.no_grad():
    for stidx in range(len(X_train_tokens)):
        tokens = X_train_tokens[stidx]
        tokens_t = torch.LongTensor(tokens)#.to(device)
        segment_t = torch.LongTensor([1] * len(tokens))#.to(device)
        outputs = bert_model(tokens_t.unsqueeze(0),segment_t.unsqueeze(0))
        embeddings = outputs[0]
        results.append(embeddings.cpu())
        #results[stidx] = embeddings.cpu()

print('me')