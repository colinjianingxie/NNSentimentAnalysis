
import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter, defaultdict
import operator
import os, math
import numpy as np
import random
import copy
from torchtext import data

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# We are using 'spacy' tokenizer. You can also write your own tokenizer. You can download spacy from
# this site https://spacy.io/usage
TEXT = data.Field(tokenize = 'spacy')
LABEL = data.LabelField(dtype = torch.float)



from torchtext import datasets

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state = random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')
print(f'Number of validation examples:{len(test_data)}')



MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)


print(TEXT.vocab.itos[:10])


BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)

"""The function binary_accuracy will be used to compute accuracy from logits"""

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

"""In Class WordEmbAvg we define our model.The model works in the following way.

1. The input to our model is a batch of sentences. All sentences are made of equal length by padding.
   Every word in the sentence is represented by one-hot encoding. So, sentence is a list of one-hot encoding.

2. In input passes through an embedding layer. The embedding layer converts the one-hot encoding for every word into a word vector.

3. We take the average of all the word vectors in a sentence. This vector is then used as an inout to a neural network.

4. The neural network has only one output. It tells you the probability of the sentence having a particular label. (We will only have two labels)

"""

class WordEmbAvg(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim, pad_idx):
        
        super().__init__()
        
        # Define embedding layer in the next layer.
        # It should be something like 
        #emb = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        
        #TODO
        self.emb = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        #Define your neural network. It can single layer or multiple layer neural network
        # You don't need apply a softmax in the output layer
        
        #TODO
        self.linear1 = nn.Linear(embedding_dim, output_dim)
        
        
    def forward(self, text):

        
        #Input goes to the embedding layer
       
        #TODO
        output = self.emb(text)
        # Take the average of all word embeddngs. Please check how to mean() on a tensor on pytorch
        
        #TODO
        output = torch.mean(output, 0)
        # Previous input now goes into the neural network
        
        #TODO
        output = self.linear1(output)
        
        
        return output

class Training_module( ):

    def __init__(self, model):
       self.model = model
    
       #The loss function should be binary cross entropy with logits. 
       self.loss_fn = nn.BCEWithLogitsLoss()
       # Choose your favorite optimizer
       self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
    
    def train_epoch(self, iterator):

        epoch_loss = 0
        epoch_acc = 0
    
    
        for batch in iterator:
        
            self.optimizer.zero_grad()
                
            #TODO

            predictions = self.model(batch.text).squeeze(1)
            #print("batch 2 working")
            #print(predictions.shape)
            loss = self.loss_fn(predictions, batch.label)
            #print("batch 3 working")
            acc = binary_accuracy(predictions, batch.label)
            #print("batch 4 working")
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    def train_model(self, train_iterator, dev_iterator):
        """
        This function processes the entire training set for multiple epochs.
        """  
        dev_accs = [0.]
        for epoch in range(6):
            self.train_epoch(train_iterator)
            dev_acc = self.evaluate(dev_iterator)
            print("dev acc: {}".format(dev_acc[1]), "dev loss:{}".format(dev_acc[0]))
            if dev_acc[1] > max(dev_accs):
                best_model = copy.deepcopy(self)
            dev_accs.append(dev_acc[1])
        return best_model.model
                
    def evaluate(self, iterator):
        '''
        This function evaluate the data with the current model.
        '''
        epoch_loss = 0
        epoch_acc = 0
    
        #model.eval()
    
        with torch.no_grad():
    
            for batch in iterator:

                predictions = self.model(batch.text).squeeze(1)
        
                loss = self.loss_fn(predictions, batch.label)
        
                acc = binary_accuracy(predictions, batch.label)
        
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

INPUT_DIM = len(TEXT.vocab)
#You can try many different embedding dimensions. Common values are 20, 32, 64, 100, 128, 512
EMBEDDING_DIM = 128
OUTPUT_DIM = 1
#Get the index of the pad token using the stoi function
# refrence from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


model = WordEmbAvg(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

model = model.to(device)
tm =Training_module(model)

#Traing the model
best_model = tm.train_model(train_iterator, valid_iterator)

tm.model = best_model
test_loss, test_acc = tm.evaluate(test_iterator)
#Accuracy on the best data. Should be possible to get accuracy around 80-85%
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')



#TODO

# Extract the weights from the linear layer 
linear_weights = tm.model.linear1.weight
embedding_mat = tm.model.emb.weight

# 1*100 * 100 * 25000
norm_vector = torch.matmul(linear_weights, embedding_mat.T)

max_values, max_indices = torch.topk(input=norm_vector, k=10, largest=True) 
min_values, min_indices = torch.topk(input=norm_vector, k=10, largest=False) 

max_word_indices = max_indices.tolist()[0]
min_word_indices = min_indices.tolist()[0]

min_words = []
max_words = []

for word in max_word_indices:
	max_words.append(TEXT.vocab.itos[word])

for word in min_word_indices:
	min_words.append(TEXT.vocab.itos[word])

print("Max words", max_words)
print("Min words", min_words)


"""

The top 10 words that are POSITIVE are: 
['great', 'best', 'perfect', 'favorite', '8/10', 'wonderful', 'excellent', 'loved', 'brilliant', 'amazing']



The top 10 words that are NEGATIVE are: 
['worst', 'bad', 'waste', 'worse', 'awful', 'lame', 'terrible', 'horrible', 'dull', 'stupid']
"""

