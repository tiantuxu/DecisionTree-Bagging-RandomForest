#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd


# In[2]:


df = pd.read_csv('dating-full.csv')
df = df.head(6500)


# In[3]:


# (i) Drop clumns
df = df.drop('race', axis=1)
df = df.drop('race_o', axis=1)
df = df.drop('field', axis=1)


# In[4]:


# (ii) gender and 1(iv)
df['gender'] = df['gender'].astype('category')
df['gender'] = df['gender'].cat.codes

preference_scores_of_participant  = ['attractive_important', 'sincere_important', 'intelligence_important', 'funny_important', 'ambition_important', 'shared_interests_important']
preference_scores_of_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',  'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']

df[preference_scores_of_participant] = df[preference_scores_of_participant].div(df[preference_scores_of_participant].sum(axis=1), axis=0)
df[preference_scores_of_partner] = df[preference_scores_of_partner].div(df[preference_scores_of_partner].sum(axis=1), axis=0)


# In[5]:


keys = df.keys()
for k in keys:
    df[k] = pd.cut(df[k], 2, labels=[0,1])


# In[6]:


#print df
df_test = df.sample(frac=0.2, random_state=47)
df_test.to_csv('testSet.csv', index=False)
# Subtract test from training
df_train = df[~df.index.isin(df_test.index)]
df_train.to_csv('trainingSet.csv', index=False)


# In[ ]:




