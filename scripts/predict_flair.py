"""
    @author : Sakshi Tantak
"""

# Imports
import re
from time import time
from flair.models import TextClassifier
from flair.data import Sentence
import emoji

MODEL_PATH = '/home/sakshi/projects/twt-sentiment-analysis/models/flair-sentiment-classifier/best-model.pt'

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

class SentimentClassifier:
    def __init__(self):
        print('Loading Flair sentiment classifier ...')
        start = time()
        self.model = TextClassifier.load(MODEL_PATH)
        print(f'Time taken to load flair sentiment classifier = {time() - start}')

    def predict(self, text):
        text = clean_text(text)
        print(f'cleaned text : {text}')
        sentence = Sentence(text)
        start = time()
        self.model.predict(sentence)
        print(f'Inference time = {time() - start}')
        if sentence.to_dict()['all labels'][0]['value'] == '1':
            return 'positive'
        return 'negative'

if __name__ == '__main__':
    text = input('Input tweet : ')
    text = clean_text(text)
    classifier = SentimentClassifier()
    prediction = classifier.predict(text)
    print(text, ' : ', prediction)