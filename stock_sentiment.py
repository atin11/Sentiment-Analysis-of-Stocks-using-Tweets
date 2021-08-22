#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import re
import flair
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns


# In[3]:


sentiment_model = flair.models.TextClassifier.load('en-sentiment')


# In[1]:


pip install yfinance


# In[ ]:


pip install p


# In[119]:


pip install seaborn


# In[4]:


def get_data(tweet):
    data = {
        'id': tweet['id'],
        'created_at': tweet['created_at'],
        'text': tweet['text']
    }
    return data

def clean(tweet):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    name = re.compile(r"(?i)@"+re.escape(params['query'])+"(?=\b)")
    user = re.compile(r"(?i)@[a-z0-9_]+")

    # we then use the sub method to replace anything matching
    tweet = whitespace.sub(' ', str(tweet))
    tweet = web_address.sub('', tweet)
    tweet = name.sub(params['query'], tweet)
    tweet = user.sub('', tweet)
    return tweet


# In[5]:


BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAFweLwEAAAAA0WIgssknoDJ37EEYZdlgXV%2FogWY%3DBDQ1RfYBK3HIFnd9Kix6LEixpYpYDOuTQpXTR6RkCcDrgkXEDT'

# setup the API request
endpoint = 'https://api.twitter.com/2/tweets/search/recent'
headers = {'authorization': 'Bearer '+ BEARER_TOKEN}
params = {
    'query': '(GME) (lang:en)',
    'max_results': '10',
    'tweet.fields': 'created_at,lang'
}

dtformat = '%Y-%m-%dT%H:%M:%SZ'  # the date format string required by twitter

# use this function to subtract 60 mins from our datetime string
def time_travel(now, mins):
    now = datetime.strptime(now, dtformat)
    back_in_time = now - timedelta(minutes=mins)
    return back_in_time.strftime(dtformat)
    
now = datetime.now()  # get the current datetime, this is our starting point
last_week = now - timedelta(days=7)  # datetime one week ago = the finish line
now = now.strftime(dtformat)  # convert now datetime to format for API

df = pd.DataFrame()  # initialize dataframe to store tweets

df = pd.DataFrame()  # initialize dataframe to store tweets
while True:
    if datetime.strptime(now, dtformat) < last_week:
        # if we have reached 7 days ago, break the loop
        break
    pre60 = time_travel(now, 60)  # get 60 minutes before 'now'
    # assign from and to datetime parameters for the API
    params['start_time'] = pre60
    params['end_time'] = now
    
    #print(response.json())
    now = pre60  # move the window 60 minutes earlier
    
    try:
        response = requests.get(endpoint,
                                params=params,
                                headers=headers)  # send the request
        for tweet in response.json()['data']:
            row = get_data(tweet)  # we defined this function earlier
            df = df.append(row, ignore_index=True)
    except:
        pass


# In[6]:


df.head()


# In[7]:


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



# In[8]:


# add probability and sentiment predictions to tweets dataframe
tweets['probability'] = probs
tweets['sentiment'] = sentiments
print(len(probs), len(positive), len(negative))


# In[9]:


pd.options.display.max_colwidth = 1000

tweets.head()


# In[ ]:





# In[ ]:





# In[17]:


# create a single negative-positive sentiment value
xax = []
sentimentf = []
sentiment = []
count = 0
yval = 0
for row in tweets.iterrows():
    if row[1]['sentiment'] == 'POSITIVE':
        # if positive we use a +ve value
        value = row[1]['probability']
    else:
        # otherwise it is negative so we should use a -ve value
        value = -row[1]['probability']
    sentiment.append(value)
    count +=1
    if count==10:
        yval = sum(sentiment)
        sentimentf.append(yval)
        xax.append(row[1]['created_at'][:-14])
        sentiment = []
        yval = 0
        count = 0

#sentimentf = [x*10 for x in sentimentf]  # multiply so we can see +ve/-ve
#stock_avg = stock_data['Close'].mean()  # get the average TSLA price
#sentimentf = [x+stock_avg for x in sentimentf]  # add the average stock price so that it will be centred when we visualize
dataf = pd.DataFrame()
#dataf['sentiments'] = sentimentf
dataf['time'] = xax
dataf['date_ordinal']=pd.to_datetime(dataf['time']).apply(lambda date: date.toordinal())


# In[18]:


dataf.head()


# In[24]:


STOCK = yf.Ticker("GME")
stock_data = STOCK.history(
    start = dataf['time'].min(),
    end = dataf['time'].max(),
    interval='60m'
).reset_index()

#sentimentf = [x*10 for x in sentimentf]  # multiply so we can see +ve/-ve
stock_avg = stock_data['Close'].mean()  # get the average TSLA price
sentimentf = [x+stock_avg for x in sentimentf]  # add the average stock price so that it will be centred when we visualize
dataf['sentiments'] = sentimentf

stockdf = pd.DataFrame()
stockdf['Datetime'] = stock_data['Datetime']
stockdf['close'] = stock_data['Close']
stockdf['x'] = [i for i in range(len(stockdf['close']))]
stockdf.head()


# In[ ]:





# In[25]:


# the stock price

sns.lineplot(x=stock_data['Datetime'], y=stock_data['Close'])


# In[26]:



# plot the sentiment data
sns.regplot(x='date_ordinal', y='sentiments', data = dataf, x_estimator = np.mean)


# In[ ]:




