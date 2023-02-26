"""
    @author : Sakshi Tantak
"""

# Imports
import re
from time import time
import emoji
from setfit import SetFitModel

MODEL_PATH = '/home/sakshi/projects/twt-sentiment-analysis/models/setfit-classifier'

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
        print('Loading SetFit sentiment classifier ...')
        start = time()
        self.model = SetFitModel.from_pretrained(MODEL_PATH)
        print(f'Time taken to load SetFit sentiment classifier = {time() - start}')

    def predict(self, text):
        text = clean_text(text)
        print(f'cleaned text : {text}')
        start = time()
        output = self.model([text])
        print(f'Inference time = {time() - start}')
        return 'positive' if output.item()==1 else 'negative'

if __name__ == '__main__':
    text = input('Input tweet : ')
    text = clean_text(text)
    classifier = SentimentClassifier()
    prediction = classifier.predict(text)
    print(text, ' : ', prediction)