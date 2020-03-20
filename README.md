# Kaggle-NLP-Disaster-Tweets
[Disaster Tweet, Real or Not?](https://www.kaggle.com/c/nlp-getting-started)  
Description: Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster.

The objective of this project is to learn different techniques associated with NLP analysis.

## Visualizing and analyzing
We start by analyzing the distribution of the target column.  
As the graph below shows, there is a roughly equal distribution of both cases and we have a good amount of data.

![realvsnotreal](https://user-images.githubusercontent.com/45593399/77186142-b6286b00-6ad2-11ea-8f1c-245331969f8b.PNG)

Example of disaster tweet:  
> There's an emergency evacuation happening now in the building across the street.

And of not a disaster tweet:  
> infected bloody ear piercings are always fun??

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

We can look further into the keywords and which are more common for disaster tweets. Here are the top and bottom 20 words sorted by times used in real disaster tweets minus times used in not disaster tweets. As can be seen, some words are really good indicators for disasters and does not occur in many other settings
![image](https://user-images.githubusercontent.com/45593399/77187698-2e902b80-6ad5-11ea-993b-5362bee069ed.png)

A quick look on the word and character distributions show that there really isn't much difference between the real disaster tweets and other tweets.
![length_analysis](https://user-images.githubusercontent.com/45593399/77188574-7f545400-6ad6-11ea-8e50-e5fda26db320.PNG)

The locations tag is tricky to utilizie for analyzation because som are very specific while others are broad. Additionally, only 30% of the tweets had locations given.
![locations](https://user-images.githubusercontent.com/45593399/77188790-d5c19280-6ad6-11ea-934f-77581f3ddd26.PNG)

When it comes to to the sentiment, there seems to be a correlation that disaster tweets have on average lower sentiment scores which makes intuitive sense. But there is a lot of spread, so we have to perform a statistical test to find out if this is significant. A one way ANOVA test gives a very low p-value which indicates a statistical significance that disaster tweets on average have lower sentiment score.

![sentiment](https://user-images.githubusercontent.com/45593399/77189730-5208a580-6ad8-11ea-9df7-f4cfb5b57776.PNG)

