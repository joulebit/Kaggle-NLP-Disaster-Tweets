import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import itertools

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Split training set into fake tweets and real tweets
train_real_disaster_tweets = train_data[train_data["target"] == 1]
train_fake_disaster_tweets = train_data[train_data["target"] == 0]

def real_fake_distribution():
        
    plt.subplot(121)
    plt.bar(['Real disaster'], [train_real_disaster_tweets.count()[1]], alpha=0.7, color='g', label='train_fake')
    plt.bar(['Not real disaster'], [train_fake_disaster_tweets.count()[1]], alpha=0.7, color='r', label='train_fake')
    #plt.hist(train_tweet_length_real, bins, alpha=0.7, color='g', label='train_real')
    plt.ylabel("number of occurences")
    plt.legend(loc='upper left')
    plt.show()


def length_histogram():

    # Add columns in Dataframe for wordcount and number of characters
    train_real_disaster_tweets["tweet_length"] = [len(sentence.split()) for sentence in train_real_disaster_tweets["text"].tolist()]
    train_fake_disaster_tweets["tweet_length"] = [len(sentence.split()) for sentence in train_fake_disaster_tweets["text"].tolist()]
    test_data["tweet_length"] = [len(sentence.split()) for sentence in test_data["text"].tolist()]

    train_real_disaster_tweets["characters"] = [len(sentence) for sentence in train_real_disaster_tweets["text"].tolist()]
    train_fake_disaster_tweets["characters"] = [len(sentence) for sentence in train_fake_disaster_tweets["text"].tolist()]

    # Make lists for more easily plotting
    train_tweet_length_real = train_real_disaster_tweets["tweet_length"].tolist()
    train_tweet_length_fake = train_fake_disaster_tweets["tweet_length"].tolist()
    train_tweet_characters_real = train_real_disaster_tweets["characters"].tolist()
    train_tweet_characters_fake = train_fake_disaster_tweets["characters"].tolist()

    # Plot histograms
    plt.subplot(121)
    bins = np.linspace(0, 32, 33)
    plt.hist(train_tweet_length_fake, bins, alpha=0.7, color='b', label='train_fake')
    plt.hist(train_tweet_length_real, bins, alpha=0.7, color='g', label='train_real')
    plt.ylabel("frequency")
    plt.xlabel("wordcount")
    plt.legend(loc='upper right')

    plt.subplot(122)
    bins = np.linspace(0, 140, 35)
    plt.hist(train_tweet_characters_fake, bins, alpha=0.7, color='b', label='train_fake')
    plt.hist(train_tweet_characters_real, bins, alpha=0.7, color='g', label='train_real')
    plt.ylabel("frequency")
    plt.xlabel("character count")
    plt.legend(loc='upper right')


    plt.show()


def sentiment():

    def clean_tweet(tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 
    
    train_real_disaster_tweets["sentiment"] = [TextBlob(clean_tweet(sentence)).sentiment.polarity for sentence in train_real_disaster_tweets["text"].tolist()]
    train_real_disaster_tweets["tweet_length"] = [len(sentence.split()) for sentence in train_real_disaster_tweets["text"].tolist()]
    train_fake_disaster_tweets["sentiment"] = [TextBlob(clean_tweet(sentence)).sentiment.polarity for sentence in train_fake_disaster_tweets["text"].tolist()]
    train_fake_disaster_tweets["tweet_length"] = [len(sentence.split()) for sentence in train_fake_disaster_tweets["text"].tolist()]


    import scipy
    print(scipy.stats.f_oneway(train_real_disaster_tweets["sentiment"],train_fake_disaster_tweets["sentiment"]))

    train_real_sentiment = train_real_disaster_tweets["sentiment"].tolist()
    train_real_length = train_real_disaster_tweets["tweet_length"].tolist()

    train_fake_sentiment = train_fake_disaster_tweets["sentiment"].tolist()
    train_fake_length = train_fake_disaster_tweets["tweet_length"].tolist()

    print("Average sentiment for disaster tweets", sum(train_real_sentiment)/len(train_real_sentiment))
    print("Average sentiment for not disaster tweets", sum(train_fake_sentiment)/len(train_fake_sentiment))
    train_real_sentiment_sum = [0]*33
    train_real_sentiment_num = [0]*33
    train_fake_sentiment_sum = [0]*33
    train_fake_sentiment_num = [0]*33

    # Calculate sum sentiment for each word length of real tweets
    for i in range(len(train_real_sentiment)):
        index = train_real_length[i]
        train_real_sentiment_sum[index] += train_real_sentiment[i]
        train_real_sentiment_num[index] += 1

    # Calculate average sentiment for each word length of real tweets
    train_real_average_sentiment = []
    for i in range(len(train_real_sentiment_num)):
        if train_real_sentiment_num[i] != 0:
            train_real_average_sentiment.append(train_real_sentiment_sum[i]/train_real_sentiment_num[i])
        else:
            train_real_average_sentiment.append(0)

    


    # Calculate sum sentiment for each word length of fake tweets
    for i in range(len(train_fake_sentiment)):
        index = train_fake_length[i]
        train_fake_sentiment_sum[index] += train_fake_sentiment[i]
        train_fake_sentiment_num[index] += 1

    # Calculate average sentiment for each word length of fake tweets
    train_fake_average_sentiment = []
    for i in range(len(train_fake_sentiment_num)):
        if train_fake_sentiment_num[i] != 0:
            train_fake_average_sentiment.append(train_fake_sentiment_sum[i]/train_fake_sentiment_num[i])
        else:
            train_fake_average_sentiment.append(0)

    bins = np.linspace(0, 32, 33)
    plt.scatter(bins, train_real_average_sentiment, s=[freq for freq in train_real_sentiment_num], label='real tweets')
    plt.scatter(bins, train_fake_average_sentiment, s=[freq for freq in train_fake_sentiment_num],label='fake tweets')
    plt.ylabel("average sentiment")
    plt.xlabel("wordcount")
    plt.legend(loc='upper right')
    plt.show()


def keyword():
    keyword_data_real = train_real_disaster_tweets["keyword"].fillna(value="[empty]")
    keyword_data_real = pd.Series(keyword_data_real).str.replace('%20', ' ')
    train_real_keywords = keyword_data_real.tolist()

    keyword_data_fake = train_fake_disaster_tweets["keyword"].fillna(value="[empty]")
    keyword_data_fake = pd.Series(keyword_data_fake).str.replace('%20', ' ')
    train_fake_keywords = keyword_data_fake.tolist()

    def CountFrequency(my_list): 
    
        # Creating an empty dictionary  
        freq = {} 
        for item in my_list: 
            if (item in freq): 
                freq[item] += 1
            else: 
                freq[item] = 1

        return freq

    frequency_real = CountFrequency(train_real_keywords)
    frequency_fake = CountFrequency(train_fake_keywords)
    diff_freq = {}

    for key in frequency_real:
        if key in frequency_fake:
            diff_freq[key] = frequency_real[key] - frequency_fake[key]
    
    sorted_frequency = {k: v for k, v in sorted(diff_freq.items(), key=lambda item: item[1], reverse=True)}
    sorted_frequency_first = dict(itertools.islice(sorted_frequency.items(), 20))
    sorted_frequency_last = dict(itertools.islice(sorted_frequency.items(), len(sorted_frequency.items()) - 20, None,))
    #sorted_frequency_first.update(sorted_frequency_last)
    #sorted_frequency_combined = sorted_frequency_first

    plt.figure(figsize=(20,10))
    plt.barh(list(sorted_frequency_first.keys()), list(sorted_frequency_first.values()), color="g")
    plt.barh(list(sorted_frequency_last.keys()), list(sorted_frequency_last.values()), color="r")
    plt.gca().invert_yaxis()
    plt.xlabel("difference in keyword occurence")
    plt.show()


def location():
    location_data_real = train_real_disaster_tweets["location"].fillna(value="")
    location_data_real = location_data_real.tolist()

    location_data_fake = train_fake_disaster_tweets["location"].fillna(value="")
    location_data_fake = location_data_fake.tolist()

    def CountFrequency(my_list): 
    
        # Creating an empty dictionary  
        freq = {} 
        for item in my_list: 
            if (item in freq): 
                freq[item] += 1
            else: 
                freq[item] = 1

        return freq

    frequency_real = CountFrequency(location_data_real)
    frequency_fake = CountFrequency(location_data_fake)


    sorted_frequency_real = {k: v for k, v in sorted(frequency_real.items(), key=lambda item: item[1], reverse=True)}
    sorted_frequency_real = dict(itertools.islice(sorted_frequency_real.items(), 1, 21))

    sorted_frequency_fake = {k: v for k, v in sorted(frequency_fake.items(), key=lambda item: item[1], reverse=True)}
    sorted_frequency_fake = dict(itertools.islice(sorted_frequency_fake.items(), 1, 21))
    

    plt.subplot(2,1,1)
    plt.barh(list(sorted_frequency_real.keys()), list(sorted_frequency_real.values()), color="g")
    plt.gca().invert_yaxis()
    plt.xlabel("given location real disaster tweets")

    plt.subplot(2,1,2)
    plt.barh(list(sorted_frequency_fake.keys()), list(sorted_frequency_fake.values()), color="r", alpha=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel("given location for not real disaster tweets")
    plt.show()


def main():
    length_histogram()
    sentiment()
    keyword()
    location()
    real_fake_distribution()
  
if __name__== "__main__":
  main()