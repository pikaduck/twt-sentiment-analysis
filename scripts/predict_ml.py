"""
    @author : Sakshi Tantak
"""

# Imports
import re
import string
import pickle
from time import time

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import emoji

# nltk.download('punkt')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('wordnet')

stops = stopwords.words('english')
negatives = ['no','nor','not','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',
  "hasn't",'haven',"haven't",'isn',"isn't",'mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",
  'wasn',"wasn't",'weren',"weren't","won't",'wouldn',"wouldn't",'don',"don't"]
stops = set([stop for stop in stops if stop not in negatives])

lemmatizer = WordNetLemmatizer()
MODEL, COUNT_VECTORIZER, TFIDF = None, None, None
MODEL_PATH = '/home/sakshi/projects/twt-sentiment-analysis/models/xgb_clf.sav'
COUNT_VECTORIZER_PATH = '/home/sakshi/projects/twt-sentiment-analysis/models/count_vectorizer.sav'
TFIDF_PATH = '/home/sakshi/projects/twt-sentiment-analysis/models/tfidf_vectorizer.sav'

def clean_text(text):
    text = re.sub(r'[\.]+', '.', text)
    # print(text)
    text = re.sub(r'[\!]+', '!', text)
    # print(text)
    text = re.sub(r'[\?]+', '!', text)
    # print(text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    # print(text)
    text = re.sub(r'@\w+', '', text).strip().lower()
    # print(text)
    text = re.sub(r'\s[n]+[o]+', ' no', text)
    # print(text)
    text = re.sub(r'n\'t', 'n not', text)
    # print(text)
    text = re.sub(r'\'nt', 'n not', text)
    # print(text)
    text = re.sub(r'\'re', ' are', text)
    # print(text)
    text = re.sub(r'\'s', ' is', text)
    # print(text)
    text = re.sub(r'\'d', ' would', text)
    # print(text)
    text = re.sub(r'\'ll', ' will', text)
    # print(text)
    text = re.sub(r'\'ve', ' have', text)
    # print(text)
    text = re.sub(r'\'m', ' am', text)
    # print(text)
    # map variations of nope to no
    text = re.sub(r'\s[n]+[o]+[p]+[e]+', ' no', text)
    # print(text)
    # clean websites mentioned in text
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\~)*\b', '', text, flags=re.MULTILINE).strip()
    # print(text)
    text = re.sub(r'(www.)(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE).strip()
    # print(text)
    text = re.sub(r'\w+.com', '', text).strip()
    # print(text)
    text = emoji.demojize(text)
    return text

def remove_punctuation(text):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(translator)
    return re.sub(r'\s+', ' ', text).strip()

def remove_numbers(text):
  return re.sub(r'[0-9]+', '', text)

def remove_stopwords_and_lemmatize(text):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens if token.strip() not in stops]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def load_model():
    global MODEL, COUNT_VECTORIZER, TFIDF

    if MODEL is None:
        with open(MODEL_PATH, 'rb') as f:
            print('Loading classifier ...')
            start = time()
            MODEL = pickle.load(f)
            print(f'Time taken to load model = {time() - start}')
        f.close()

    if COUNT_VECTORIZER is None:
        with open(COUNT_VECTORIZER_PATH, 'rb') as f:
            print('Loading count vectorizer ...')
            start = time()
            COUNT_VECTORIZER = pickle.load(f)
            print(f'Time taken to load count vectorizer = {time() - start}')
        f.close()

    if TFIDF is None:
        with open(TFIDF_PATH, 'rb') as f:
            print('Loading tfidf vectorizer ...')
            start = time()
            TFIDF = pickle.load(f)
            print(f'Time taken to load tfidf vectorizer = {time() - start}')
        f.close()

def predict(text):
    if MODEL is None:
        load_model()

    text = clean_text(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_stopwords_and_lemmatize(text)

    vector = COUNT_VECTORIZER.transform([text]).toarray()
    vector = TFIDF.transform(vector).toarray()
    start = time()
    prediction = MODEL.predict(vector).item()
    print(f'Inference time = {time() - start}')
    return 'positive' if prediction == 1 else 'negative'

if __name__ == '__main__':
    text = input('Enter tweet')
    # text = "i am so bored!!!"
    prediction = predict(text)
    print(text, ' : ', prediction)