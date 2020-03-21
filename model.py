import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Clean textfield
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Applying the cleaning function to both test and training datasets
train['text'] = train['text'].apply(lambda x: clean_text(x))
test['text'] = test['text'].apply(lambda x: clean_text(x))


# Tokenizing the training and the test set
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))
test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))


stop_words = stopwords.words('english')

def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    words = [w for w in text if w not in stop_words]
    return words


train['text'] = train['text'].apply(lambda x : remove_stopwords(x))
test['text'] = test['text'].apply(lambda x : remove_stopwords(x))


# Combine tokenized text
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

train['text'] = train['text'].apply(lambda x : combine_text(x))
test['text'] = test['text'].apply(lambda x : combine_text(x))


# Add keyword and location to text to be vectorized
train["keyword"] = train["keyword"].fillna("")
train["location"] = train["location"].fillna("")
train["text"] = train["keyword"].astype(str) + " " + train["location"].astype(str) + " " + train["text"]

test["keyword"] = test["keyword"].fillna("")
test["location"] = test["location"].fillna("")
test["text"] = test["keyword"].astype(str) + " " + test["location"].astype(str) + " " + test["text"]

# Sentiment
"""
train["sentiment"] = [TextBlob(sentence).sentiment.polarity for sentence in train["text"].tolist()]
test["sentiment"] = [TextBlob(sentence).sentiment.polarity for sentence in test["text"].tolist()]
"""

# Vectorization
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train["text"])
test_vectors = count_vectorizer.transform(test["text"])

# Use classifier
clf = linear_model.LogisticRegression(C=0.5)
scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")
print(scores)
clf.fit(train_vectors, train["target"])

# Save submission to file
submission = pd.DataFrame(test["id"].tolist(),columns=["Id"])
submission["target"] = clf.predict(test_vectors)
submission.to_csv("submission.csv", index=False)
