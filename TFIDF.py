#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import pickle as pkl
import math
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import gc

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


article_dir = "./articles"
path_list = [os.path.join(article_dir, x) for x in os.listdir(article_dir)]


# In[3]:


len(path_list)


# In[72]:


path_list[0]


# In[73]:


corpus = []
article_ids = []
for p in path_list:
    with open(p, 'rb') as f:
        txt = pkl.load(f)
        if len(txt)>5000:
            corpus.append(txt)
            article_ids.append(p.split('/')[2].split('.')[0])


# In[74]:


len(corpus)


# In[76]:


int(article_ids[0])


# ## IDF processing

# In[6]:


num_docs = len(corpus)


# In[7]:


v = CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english',min_df=1)
vf = v.fit_transform(corpus)


# In[8]:


terms = v.get_feature_names()
freqs = np.asarray(vf.sum(axis=0).ravel())[0]


# In[9]:


idfs = [math.log10(num_docs/f) for f in freqs]


# In[10]:


custom_weight1 = np.array([1+math.log10(f) for f in freqs])


# In[52]:


#custom_weight2 = np.array([1+math.log10(f)*0.1 for f in freqs])


# In[11]:


custom_weight1[0:10]


# In[54]:


#custom_weight2[0:10]


# ## straight into TF/IDF

# In[ ]:


#interval = 500
#cur_index = 0
#while cur_index<=num_docs:
#    v = CountVectorizer(ngram_range=(1,1), stop_words='english', vocabulary=terms. min_df=1)
#    vf = v.fit_transform(corpus[cur_index:cur_index+interval])
    
    

#for p in path_list[:500]:
#    with open(p, 'rb') as f:
#        txt = pkl.load(f)
#        corpus.append(txt)

#subv = CountVectorizer(ngram_range=(1,2), stop_words='english', vocabulary=terms, min_df=1)
#subvf = v.fit_transform(corpus)


# keywords = []
# interval = 100
# cur_index = 0
# while cur_index<len(corpus):
#     # print(cur_index)
#     v = CountVectorizer(ngram_range=(1,2), stop_words='english', vocabulary=terms, min_df=1)
#     vf = v.fit_transform(corpus[cur_index:cur_index+interval])
#     a = vf.toarray()
#     
#     for article in a:
#         x = np.copy((np.log10(article+1)*idfs).argsort()[-20:][::-1])
#         keywords.append(x)
#     
#     print(len(keywords))
#     del v
#     del vf
#     del a
#     gc.collect()
#     #a_list.append(vf.toarray())
#     cur_index += interval
# 
# print(len(keywords))

# In[12]:


keywords1 = []
interval = 100
cur_index = 0
while cur_index<len(corpus):
    # print(cur_index)
    v = CountVectorizer(ngram_range=(1,2), stop_words='english', vocabulary=terms, min_df=1)
    vf = v.fit_transform(corpus[cur_index:cur_index+interval])
    a = vf.toarray()
    
    for article in a:
        x = np.copy((np.log10(article+1)*idfs*custom_weight1).argsort()[-20:][::-1])
        keywords1.append(x)
    
    print(len(keywords1))
    del v
    del vf
    del a
    gc.collect()
    #a_list.append(vf.toarray())
    cur_index += interval

print(len(keywords1))


# keywords2 = []
# interval = 100
# cur_index = 0
# while cur_index<len(corpus):
#     # print(cur_index)
#     v = CountVectorizer(ngram_range=(1,2), stop_words='english', vocabulary=terms, min_df=1)
#     vf = v.fit_transform(corpus[cur_index:cur_index+interval])
#     a = vf.toarray()
#     
#     for article in a:
#         x = np.copy((np.log10(article+1)*idfs*custom_weight2).argsort()[-20:][::-1])
#         keywords2.append(x)
#     
#     print(len(keywords2))
#     del v
#     del vf
#     del a
#     gc.collect()
#     #a_list.append(vf.toarray())
#     cur_index += interval
# 
# print(len(keywords2))

# In[78]:


#for k in keywords[2:3]:
#    output = [terms[i] for i in k]
#    print(output)


# In[13]:


for k in keywords1[2:3]:
    output = [terms[i] for i in k]
    print(output)


# In[80]:


#for k in keywords2[2:3]:
#    output = [terms[i] for i in k]
#    print(output)


# In[14]:


#x = np.array(keywords).flatten()
x1 = np.array(keywords1).flatten()
#x2 = np.array(keywords2).flatten()
#unique_x = set(x)
unique_x1 = set(x1)
#unique_x2 = set(x2)
#print(len(x))
#print(len(unique_x))
print(len(unique_x1))
#print(len(unique_x2))


# In[15]:


dummy = x1.copy().tolist()
for i in unique_x1:
    dummy.remove(i)


# In[16]:


print(len(set(dummy)))


# In[17]:


for word in list(set(dummy))[0:100]:
    print(terms[word])


# In[88]:


occ = Counter(dummy)


# In[89]:


print(len(occ.keys()))
print(len(dummy))


# In[90]:


for wid in list(occ.keys()):
    if occ[wid] < 15:
        del occ[wid]


# In[91]:


print(len(occ.keys()))


# In[92]:


for wid in list(occ.keys()):
    print(terms[wid])


# In[93]:


survived = {}
for word in list(occ.keys()):
    for i, k in enumerate(keywords1):
        if word in k:
            if article_ids[i] not in survived:
                survived[article_ids[i]] = [word]
            else:
                survived[article_ids[i]].append(word)
            #print(i,":",word)
            
print(len(survived.keys()))


# In[94]:


survived_articles = list(survived.keys())
survived_articles.sort()
for s in survived_articles[:30]:
    print(s)
    output = [terms[x] for x in survived[s]]
    print(output)


# In[95]:


len(survived_articles)


# In[35]:


print(len(survived))


# In[37]:


list(survived)[0:100]

