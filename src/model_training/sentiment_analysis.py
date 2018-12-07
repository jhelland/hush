
# coding: utf-8

# ## Sentiment analysis of n-grams in toxic comments dataset
# 
# We train a linear classifier using the "toxic" category as a reference. That is, phrases appearing in the toxic category are considered to be negative.
# 
# TF-IDF is a way of assigning weights to phrases in documents. In particular, it assigns weights according to frequency of occurence within documents -- the more often a phrase appears, the more diminished the weight. Conversely, the rarer a phrase is, the higher the weight. That is, we treat rare phrases with special attention.

# In[6]:


import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', labelsize=22)

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# In[10]:


def plot_words_polarity(df, tfidf, title=""):
    lr = LogisticRegression()
    p = make_pipeline(tfidf, lr)
    p.fit(df["comment_text"].values, df["toxic"].values)
    
    rev = sorted({v: k for k,v in p.steps[0][1].vocabulary_.items()}.items())
    polarity = pd.DataFrame({"coef": p.steps[1][1].coef_[0]},
                           index=[i[1] for i in rev]).sort_values("coef")
    
#     smallest = polarity.iloc[(polarity["coef"]).abs().argsort()[:40]]
    
    plt.figure(figsize=(20,10))
    ax = plt.subplot(1,2,1)
    polarity.tail(40).plot(kind="barh", color="red", ax=ax)
    ax = plt.subplot(1,2,2)
    polarity.head(40).plot(kind="barh", color="blue", ax=ax)
#     ax = plt.subplot(1,3,3)
#     smallest.sort_values("coef").plot(kind="barh", color="blue", ax=ax)
    plt.tight_layout()
    plt.savefig("./figures/sentiment.png", dpi=300)
    plt.show()


# In[11]:


df = pd.read_csv("../data/toxic_comments/train.csv").fillna("fillna")


# In[12]:


tfidf = TfidfVectorizer(lowercase=True, max_features=50000)
plot_words_polarity(df, tfidf, title="n-gram range (1,1)")


# In[14]:


tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(2,2))
plot_words_polarity(df, tfidf)


# In[15]:


tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(3,3))
plot_words_polarity(df, tfidf)


# In[23]:


tfidf = TfidfVectorizer(lowercase=True, max_features=50000)
plot_words_polarity(df, tfidf, title="n-gram range (1,3)")


# In[16]:


tfidf = TfidfVectorizer(lowercase=True, analyzer="char")
plot_words_polarity(df, tfidf)


