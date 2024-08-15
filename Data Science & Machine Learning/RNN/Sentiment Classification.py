#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import torch
import sys
import string
from collections import Counter


# In[3]:


vocab = {}


# In[5]:


def addToken(token):
    if token in vocab['t_2_i']:
        idx =  vocab['t_2_i'][token]
    else:
        idx = len(vocab['t_2_i'])
        vocab['t_2_i'][token] = idx
        vocab['i_2_t'][idx] = token
    return idx


# In[6]:


def initializeVocabulary():
    unkToken = '<UNK>'
    vocab['t_2_i'] = {}
    vocab['i_2_t'] = {}
    idx = addToken(unkToken)
    vocab['addUnk'] = True
    vocab['unkToken'] = unkToken
    vocab['unkTokenIdx'] = idx


# In[7]:


def addManyTokens(tokens):
    idexs = [addToken(token) for token in tokens]
    return idexs


# In[8]:


def lookuptoken(token):
    if vocab['unkTokenIdx'] >= 0:
        return vocab['t_2_i'].get(token,vocab['unkTokenIdx'])
    else:
        return vocab['t_2_i'][token]


# In[9]:


def lookupidx(idx):
    if idx not in vocab['i_2_t']:
        raise keyError("the index (%d) is not there" %(idx))
    return vocab['i_2_t'][idx]


# In[10]:


# if cutoff in not more than 25 than don't add in vocabulary
def vocabularyFromDataFrame(df,cutoff=25):
    initializeVocabulary()
    wordCounts = Counter()
    for i in df.review:
        for word in i.split(" "):
            if word not in string.punctuation:
                wordCounts[word] += 1
    for word,count in wordCounts.items():
        if count > cutoff:
            addToken(word)


# In[11]:


df = pd.read_csv(r"C:\Users\Sub\CodeSpace\AI\mastering_recurrent_neural_networks\Data\reviews.csv")


# In[12]:


vocabularyFromDataFrame(df)


# In[17]:


lookuptoken('this')


# In[18]:


lookupidx(128)


# In[ ]:




