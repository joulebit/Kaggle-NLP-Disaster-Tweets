# Kaggle-NLP-Disaster-Tweets
[Disaster Tweet, Real or Not?](https://www.kaggle.com/c/nlp-getting-started)  
Description: Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster.

The objective of this project is to learn different techniques associated with NLP analysis.

## Visualizing and analyzing
We start by analyzing the distribution of the target column.  
As the graph below shows, there is a roughly equal distribution of both cases and we have a good amount of both real and not real disaster tweets, so that the model we get can have more significant results.

![realvsnotreal](https://user-images.githubusercontent.com/45593399/77186142-b6286b00-6ad2-11ea-8f1c-245331969f8b.PNG)

Example of disaster tweet:  
> There's an emergency evacuation happening now in the building across the street.

And example of not a disaster tweet:  
> infected bloody ear piercings are always fun??

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

We can look further into the keywords and which are more common for disaster tweets. Here are the top and bottom 20 words sorted by times used in real disaster tweets minus times used in not disaster tweets. As can be seen, some words are really good indicators for disasters and does not occur in many other settings. While other words such as "ruined" or "wrecked" are being used in many non-emergency tweets.
![image](https://user-images.githubusercontent.com/45593399/77187698-2e902b80-6ad5-11ea-993b-5362bee069ed.png)

A quick look on the word and character distributions show that there really isn't much difference in the given distributions between real disaster tweets and other tweets.
![length_analysis](https://user-images.githubusercontent.com/45593399/77188574-7f545400-6ad6-11ea-8e50-e5fda26db320.PNG)

The locations tag is tricky to utilizie in analysis because some locations are very specific while others are broad. There is not a standardized list to choose from, users put in location themselves. Additionally, only 30% of the tweets had locations given.
![locations](https://user-images.githubusercontent.com/45593399/77225785-bed07e00-6b72-11ea-98d9-0621a3c9b0dc.PNG)

When it comes to to the sentiment, there seems to be a correlation that disaster tweets have on average lower sentiment scores which intuitively makes sense. But there is a lot of spread, so we have to perform a statistical test to find out if this is significant. A one way ANOVA test gives a very low p-value which indicates a statistical significance that disaster tweets on average have lower sentiment score. the size of the bubbles indicate the frequency of tweets with the given word count.

![sentiment](https://user-images.githubusercontent.com/45593399/77189730-5208a580-6ad8-11ea-9df7-f4cfb5b57776.PNG)



## Data cleaning
> “A machine learning model is only as good as the data it is fed”

First step is to clean the tweet's textfield to get it more standardized. Which means making it all lower case, removing links, numbers and punctuations. This is done so that the algorithm does not treat the same words in different cases as different.
> "Does it really matter!,Deaths 3 http://t.co/nApviyGKYK"  --> "does it really matter deaths"

Another measure that was taken was stop word removal. Stop words are extremely common words which would appear to be of little value, for example "and", "the", "a", "an", and similar words. To help identify stop words, a word tokenizer is used.


## Execution
Keywords and locations were added together with the tweet text. Then a bag of words representation was used to transform text into a meaningful vector.
### Bag of Words
The bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:

* A vocabulary of known words.
* A measure of the presence of known words.

Why is it is called a “bag” of words? That is because any information about the order or structure of words in the document is discarded and the model is only concerned with whether the known words occur in the document, not where they occur in the document.

For example, ![image](https://user-images.githubusercontent.com/45593399/77227941-2ba04400-6b84-11ea-95e1-52f5bb3aacc6.png)

source: [Natural Language Processing course on coursera](https://www.coursera.org/learn/language-processing)

We can do this using scikit-learn's CountVectorizer, where every row will represent a different tweet and every column will represent a different word.

The model chosen was a simple Logistic regression with a Inverse of regularization strength of 0.5 to specify stronger regularization.


## Results
Almost 80% of new tweets were correctly categorized. This put my solution in the 54th percentile among other users. The dataset got leaked, so the best noncheated solution was [this one](https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert) with 83.6% of tweets correctly categorized. Even with no data cleaning and only word vectorization on the raw tweets, users could get a score in the low-middle 70%.  
![image](https://user-images.githubusercontent.com/45593399/77225921-2509d080-6b74-11ea-938d-99540b9f5911.png)

## Improvements
The data cleaning process could be improved by performing two additional tasks:
* Stemming: Stemming is the process of reducing inflected (or sometimes derived) words to their stem, base or root form — generally a written word form. Example if we were to stem the following words: “Stems”, “Stemming”, “Stemmed”, “and Stemtization”, the result would be a single word “stem”.
* Lemmatization: A slight variant of stemming is lemmatization. The major difference between these is, that, stemming can often create non-existent words, whereas lemmas are actual words. So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary, but you can look up a lemma. Examples of Lemmatization are that “run” is a base form for words like “running” or “ran” or that the word “better” and “good” are in the same lemma so they are considered the same.

&nbsp;
&nbsp;
&nbsp;
&nbsp;


As the data visualization showed, sentiment would be a good feature to use since it was able to help differentiate real disaster tweets from other tweets. I was unfortunately not able to implement it at the current time.

The top scorers found great success by using Google's Bidirectional Encoder Representations from Transformers (BERT) technique from 2018. Another viable option would be TFIDF, short for term frequency–inverse document frequency, which is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. 
