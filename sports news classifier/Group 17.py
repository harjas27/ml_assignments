# -*- coding: utf-8 -*-

import nltk
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import json
import datetime
import pandas as pd
stemmer = LancasterStemmer()

#%%
# 3 classes of training data
training_data = []
path='C:/Users/dell pc/Desktop/deepl/bbcsport'

s=path
s+='/athletics/0'
for i in range(99):
    r=s
    if i<9 :
        r+='0'
        r+=str(i+1)
        r+='.txt'
    else:
        r+=str(i+1)
        r+='.txt'
    f=open(r,'r')
    training_data.append({"class":"athletics", "sentence":f.read()})   

s=path
s+='/cricket/0'
for i in range(99):
    r=s
    if i<9 :
        r+='0'
        r+=str(i+1)
        r+='.txt'
    else:
        r+=str(i+1)
        r+='.txt'
    f=open(r,'r')
    training_data.append({"class":"cricket", "sentence":f.read()})

s=path
s+='/football/0'
for i in range(99):
    r=s
    if i<9 :
        r+='0'
        r+=str(i+1)
        r+='.txt'
    else:
        r+=str(i+1)
        r+='.txt'
    f=open(r,'r')
    training_data.append({"class":"football", "sentence":f.read()})
    
s=path    
s+='/tennis/0'
for i in range(99):
    r=s
    if i<9 :
        r+='0'
        r+=str(i+1)
        r+='.txt'
    else:
        r+=str(i+1)
        r+='.txt'
    f=open(r,'r')
    training_data.append({"class":"tennis", "sentence":f.read()})

print ("%s sentences in training data" % len(training_data))

#%%
words = []
classes = []
documents = []
ignore_words = ['?']
for pattern in training_data:
    w = nltk.word_tokenize(pattern['sentence'])
    txt = " ".join([w for w in nltk.word_tokenize(pattern['sentence'])])
    words.extend(w)
    documents.append((w, pattern['class']))
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

classes = list(set(classes))

#print (len(documents), "documents")
#print (len(classes), "classes", classes)
#print (len(words), "unique stemmed words", words)

#%%
# create our training data
training = []
output = []
output_empty = [0] * len(classes)
#%%
#tfidf
t_count=[]
count=[0 for i in range(len(words))]
doc=0
while doc < len(documents):
    pattern_words = documents[doc][0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    i=0
    lis=[0 for i in range(len(words))]
    while i < len(words):
        for j in pattern_words:
            if words[i]==j:
                count[i]+=1
                lis[i]+=1
        i+=1
        t_count.append(lis)
    doc+=1
    #print(doc)
#print (t_count[0])  
#print(count)
ratio=[]  
for i in range(len(documents)):
    lis=[0 for i in range(len(words))]
    for j in range(len(words)):
        lis[j]=t_count[i][j]/count[j]
    ratio.append(lis)
#(ratio[0])    
#%%
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
#print ([stemmer.stem(word.lower()) for word in w])
#print (training[i])
#print (output[i])
#%%
import numpy as np
import time

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    l0 = x
    l1 = sigmoid(np.dot(l0, weight_0))
    l2 = sigmoid(np.dot(l1, weight_1))
    l3 = sigmoid(np.dot(l2, weight_2))
    
    return l3

#%%
def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    
    weight_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    weight_1 = 2*np.random.random((hidden_neurons, hidden_neurons)) - 1
    weight_2 = 2*np.random.random((hidden_neurons, len(classes))) - 1
 
    prev_weight_0_weight_update = np.zeros_like(weight_0)
    prev_weight_1_weight_update = np.zeros_like(weight_1)
    prev_weight_2_weight_update = np.zeros_like(weight_2)

    weight_0_direction_count = np.zeros_like(weight_0)
    weight_1_direction_count = np.zeros_like(weight_1)
    weight_2_direction_count = np.zeros_like(weight_2)
        
    for j in iter(range(epochs+1)):

        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, weight_0))
        layer_2 = sigmoid(np.dot(layer_1, weight_1))
                
        layer_3 = sigmoid(np.dot(layer_2, weight_2))

        layer_3_error = y - layer_3

        if (j% 10000) == 0 and j > 5000:
            if np.mean(np.abs(layer_3_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_3_error))) )
                last_mean_error = np.mean(np.abs(layer_3_error))
            else:
                print ("break:", np.mean(np.abs(layer_3_error)), ">", last_mean_error )
                break
                
        layer_3_delta = layer_3_error * sigmoid_output_to_derivative(layer_3)

        layer_2_error = layer_3_delta.dot(weight_2.T)
        
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(weight_1.T)

        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        weight_2_weight_update = (layer_2.T.dot(layer_3_delta))
        weight_1_weight_update = (layer_1.T.dot(layer_2_delta))
        weight_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            weight_0_direction_count += np.abs(((weight_0_weight_update > 0)+0) - ((prev_weight_0_weight_update > 0) + 0))
            weight_1_direction_count += np.abs(((weight_1_weight_update > 0)+0) - ((prev_weight_1_weight_update > 0) + 0))
            weight_2_direction_count += np.abs(((weight_2_weight_update > 0)+0) - ((prev_weight_2_weight_update > 0) + 0))
        
        weight_2 += alpha * weight_2_weight_update
        weight_1 += alpha * weight_1_weight_update
        weight_0 += alpha * weight_0_weight_update
        
        prev_weight_0_weight_update = weight_0_weight_update
        prev_weight_1_weight_update = weight_1_weight_update
        prev_weight_2_weight_update = weight_2_weight_update

    now = datetime.datetime.now()

    weight = {'weight0': weight_0.tolist(), 'weight1': weight_1.tolist(),'weight2': weight_2.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    weight_file = "weights_deep2.json"

    with open(weight_file, 'w') as outfile:
        json.dump(weight, outfile, indent=4, sort_keys=True)
    print ("saved weights to:", weight_file)
    
#%%
X = np.array(training)
y = np.array(output)
#%%

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")    

#%%
ERROR_THRESHOLD = 0.2
weight_file = 'weights_deep2.json' 
with open(weight_file) as data_file: 
    weight = json.load(data_file) 
    weight_0 = np.asarray(weight['weight0']) 
    weight_1 = np.asarray(weight['weight1'])
    weight_2 = np.asarray(weight['weight2'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results

#%%
classify("Britain's Mo Farah misses out as world champions Mutaz Essa Barshim and Nafissatou Thiam are named the IAAF's athletes of the year.")
classify("Mohamed Salah has nothing to prove when he faces his former employers Chelsea in the Premier League this weekend, his manager Juergen Klopp said on Friday.")
classify("Roger Federer beats Rafael Nadal in 5 sets to win Australian Open 2017")
classify("Ravindra Jadeja took 3 wickets in the day while R Ashwin took 4 as India bowled out Sri Lanka for 205.")
classify("Manchester United manager Jose Mourinho fears that midfielder Marouane Fellaini will leave the Premier League club at the end of the season.")
print()
