"""
    @author : Sakshi Tantak
"""

# Imports
import os
import json
import pickle
import re
import string
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import dask.dataframe as ddf
import emoji

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

stops = stopwords.words('english')
negatives = ['no','nor','not','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',
  "hasn't",'haven',"haven't",'isn',"isn't",'mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",
  'wasn',"wasn't",'weren',"weren't","won't",'wouldn',"wouldn't",'don',"don't"]
stops = set([stop for stop in stops if stop not in negatives])

lemmatizer = WordNetLemmatizer()

TRAIN = os.path.join('/home/sakshi/projects/twt-sentiment-analysis/data/processed_data', 'train.csv')
TEST = os.path.join('/home/sakshi/projects/twt-sentiment-analysis/data/processed_data', 'test.csv')

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

def load_data_and_prepare_for_training(save_path):
    df_train = pd.read_csv(TRAIN)
    df_test = pd.read_csv(TEST)
    df_test = df_test.loc[df_test['sentiment'] != 2]

    df_train['tweet_len'] = df_train['lemmatized_stop_removed_tweet'].str.len()

    df_train0 = df_train.loc[df_train['sentiment'] == 0]
    df_train4 = df_train.loc[df_train['sentiment'] == 4]

    df_train0_below_median = df_train0.loc[df_train0['tweet_len'] < df_train0['tweet_len'].median()]
    df_train0_greater_median = df_train0.loc[df_train0['tweet_len'] > df_train0['tweet_len'].median()]

    df_train4_below_median = df_train4.loc[df_train4['tweet_len'] < df_train4['tweet_len'].median()]
    df_train4_greater_median = df_train4.loc[df_train4['tweet_len'] > df_train4['tweet_len'].median()]

    df_train0_below_median = df_train0.drop_duplicates(subset = ['username'])
    df_train0_greater_median = df_train0_greater_median.drop_duplicates(subset = ['username'])

    df_train4_below_median = df_train4.drop_duplicates(subset = ['username'])
    df_train4_greater_median = df_train4_greater_median.drop_duplicates(subset = ['username'])

    df_train0_below_median = df_train0_below_median.sample(frac = 1).iloc[:500]
    df_train0_greater_median = df_train0_greater_median.sample(frac = 1).iloc[:500]

    df_train4_below_median = df_train4_below_median.sample(frac = 1).iloc[:500]
    df_train4_greater_median = df_train4_greater_median.sample(frac = 1).iloc[:500]

    df_train = pd.concat([df_train0_below_median, df_train0_greater_median, df_train4_below_median, df_train4_greater_median])

    df_train = df_train.dropna(subset = ['lemmatized_stop_removed_tweet'])
    df_test = df_test.dropna(subset = ['lemmatized_stop_removed_tweet'])

    count_vectorizer = CountVectorizer()
    tfidf = TfidfTransformer(use_idf = True, norm = 'l2', smooth_idf = True)
    save_model(count_vectorizer, save_path, 'count_vectorizer')
    save_model(tfidf, save_path, 'tfidf')

    X_train = count_vectorizer.fit_transform(df_train['lemmatized_stop_removed_tweet']).toarray()
    X_train = tfidf.fit_transform(X_train).toarray()

    # it is a binary classification problem since 2 classes exist, so mapping 4 to 1 would make sense
    y_train = df_train['sentiment'].replace([4], [1]).to_numpy()

    X_test = count_vectorizer.transform(df_test['lemmatized_stop_removed_tweet']).toarray()
    X_test = tfidf.transform(X_test).toarray()

    y_test = df_test['sentiment'].replace([4], [1]).to_numpy()

    return X_train, y_train, X_test, y_test

def train(model, X_train, y_train):
    return model.fit(X_train, y_train)

def save_model(model, dirpath, model_name):
    with open(os.path.join(dirpath, model_name + '.sav'), 'wb') as f:
        pickle.dump(model, f)
    f.close()

def make_predictions(model, X_test, y_test, dirpath, model_name):
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)
    clf_report = classification_report(y_test, predictions, output_dict = True)
    acc = accuracy_score(y_test, predictions)

    print(cm)
    print(clf_report)
    print(acc)

    with open(os.path.join(dirpath, model_name + '.report'), 'w') as f:
        json.dump(clf_report, f, indent = 4)
    f.close()


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression

    X_train, y_train, X_test, y_test = load_data_and_prepare_for_training('/home/sakshi/projects/twt-sentiment-analysis/models')
    dirpath = '/home/sakshi/projects/twt-sentiment-analysis/models'

    model = LogisticRegression()
    model = train(model, X_train, y_train)
    save_model(model, dirpath, 'logistic_regression')
    make_predictions(model, X_test, y_test, dirpath, 'logistic_regression')

    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    model = train(model, X_train, y_train)
    save_model(model, dirpath, 'decision_tree')
    make_predictions(model, X_test, y_test, dirpath, 'decision_tree')

    # train random forest

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()

    model = train(model, X_train, y_train)
    save_model(model, dirpath, 'random_forest')
    make_predictions(model, X_test, y_test, dirpath, 'random_forest')

    # train bernoulli nb since features are discrete

    from sklearn.naive_bayes import BernoulliNB

    model = BernoulliNB()

    model = train(model, X_train, y_train)
    save_model(model, dirpath, 'b_nb')
    make_predictions(model, X_test, y_test, dirpath, 'b_nb')

    # try multinomialNB

    from sklearn.naive_bayes import MultinomialNB

    model = MultinomialNB()

    model = train(model, X_train, y_train)
    save_model(model, dirpath, 'm_nb')
    make_predictions(model, X_test, y_test, dirpath, 'm_nb')

    import xgboost as xgb

    model = xgb.XGBClassifier(objective = 'binary:logistic', random_state = 42)

    model = train(model, X_train, y_train)
    save_model(model, dirpath, 'xgb_clf')
    make_predictions(model, X_test, y_test, dirpath, 'xgb_clf')

    from sklearn.ensemble import AdaBoostClassifier

    model = AdaBoostClassifier()

    model = train(model, X_train, y_train)
    save_model(model, dirpath, 'adaboost_clf')
    make_predictions(model, X_test, y_test, dirpath, 'adaboost_clf')

