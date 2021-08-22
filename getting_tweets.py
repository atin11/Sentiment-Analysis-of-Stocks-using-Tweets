#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Hello World")


# In[2]:


pip install requests


# In[3]:


conda install pytorch torchvision torchaudio -c pytorch


# In[3]:


pip install numpy
pip install pandas
pip install matplotlib
pip install flair


# In[5]:


pip install numpy


# In[6]:


pip install matplotlib


# In[7]:


pip install flair


# In[19]:


pip install --upgrade pandas


# In[20]:


import requests
import re
import flair
import matplotlib
import pandas as pd


# In[79]:


params = {
    'q': 'TENET',
    'tweet_mode': 'extended',
    'lang': 'en',
    'count': '100'
}
response = requests.get(
    'https://api.twitter.com/1.1/search/tweets.json',
    params=params,
    headers={'authorization': 'Bearer '+'AAAAAAAAAAAAAAAAAAAAAFweLwEAAAAA0WIgssknoDJ37EEYZdlgXV%2FogWY%3DBDQ1RfYBK3HIFnd9Kix6LEixpYpYDOuTQpXTR6RkCcDrgkXEDT'}
)


# In[80]:


def get_data(tweet):
    data = {
        'id': tweet['id_str'],
        'created_at': tweet['created_at'],
        'text': tweet['full_text']
    }
    return data

df = pd.DataFrame()
for tweet in response.json()['statuses']:
    row = get_data(tweet)
    df = df.append(row, ignore_index=True)


# In[23]:


pip install pandas


# In[81]:


df.head()


# In[46]:


sentiment_model = flair.models.TextClassifier.load('en-sentiment')


# In[ ]:





# In[ ]:





# In[82]:


def clean(tweet):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    tesla = re.compile(r"(?i)@"+re.escape(params['q'])+"(?=\b)")
    user = re.compile(r"(?i)@[a-z0-9_]+")

    # we then use the sub method to replace anything matching
    tweet = whitespace.sub(' ', str(tweet))
    tweet = web_address.sub('', tweet)
    tweet = tesla.sub(params['q'], tweet)
    tweet = user.sub('', tweet)
    return tweet


# In[86]:


probs = []
sentiments = []
positive = []
negative = []
tweets = df
# use regex expressions (in clean function) to clean tweets
tweets['text'] = tweets['text'].apply(clean)

for tweet in tweets['text'].to_list():
    # make prediction
    sentence = flair.data.Sentence(tweet)
    sentiment_model.predict(sentence)
    # extract sentiment prediction
    probs.append(sentence.labels[0].score)  # numerical score 0-1
    sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'
    if sentence.labels[0].value == "POSITIVE":
        positive.append(sentence.labels[0].score)
    else:
        negative.append(sentence.labels[0].score)


# add probability and sentiment predictions to tweets dataframe
tweets['probability'] = probs
tweets['sentiment'] = sentiments
print(len(probs), len(positive), len(negative))


# In[16]:


pip install --upgrade pandas


# In[87]:


#tweets.head()
pd.options.display.max_colwidth = 1000
#pd.options.display.max_rowheight
#tweets.iloc[3,2]


# In[89]:


tweets.iloc[3,2]


# In[ ]:




