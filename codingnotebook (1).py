#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# In[3]:


ps= PorterStemmer()


# In[4]:


news = pd.read_csv('train.csv')


# In[5]:


news.head()


# In[6]:


news.shape


# In[7]:


news.isnull().sum()


# In[8]:


news.dropna(how='any',inplace=True)


# In[9]:


news.shape


# In[10]:


news['label'].value_counts()


# In[11]:


sns.countplot(x=news['label'])
plt.grid(True)
plt.show()


# In[12]:


news.head()


# In[13]:


news.reset_index(drop=True, inplace=True)


# In[14]:


news.drop('id',inplace=True,axis=1)


# In[15]:


news.head(10)


# In[16]:


def clean_title(lists):
    lists=re.sub('[^a-zA-Z]',' ',lists)
    lists = lists.lower()
    lists = lists.split(' ')
    lists=[ps.stem(word) for word in lists if word not in stopwords.words('english')]
    lists= ' '.join(lists)
    return lists    


# In[18]:


corpus=[]
for i in news['title']:
    corpus.append(clean_title(i))    


# In[19]:


len(corpus)


# In[20]:


corpusdf=pd.DataFrame(corpus)


# In[21]:


corpusdf


# In[158]:


corpusdf.to_csv('title_corpus1.csv',index=False)


# In[159]:


title_corpus1= pd.read_csv('title_corpus1.csv')


# In[160]:


title_corpus1


# In[161]:


corpus1=[]
for i in title_corpus1.values:    
    for j in i:
        corpus1.append(j)


# In[162]:


len(corpus1)


# In[163]:


corpusdf.to_csv('title_corpus2.csv',index=True)


# In[164]:


title_corpus2= pd.read_csv('title_corpus2.csv')


# In[165]:


title_corpus2.drop('Unnamed: 0',inplace=True,axis=1)


# In[166]:


title_corpus2


# In[167]:


corpus2=[]
for i in title_corpus2.values:    
    for j in i:
        corpus2.append(j)


# In[168]:


len(corpus2)


# In[ ]:





# In[ ]:





# In[ ]:




