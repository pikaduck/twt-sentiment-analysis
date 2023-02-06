"""
    @author : Sakshi Tantak
"""

# Imports
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

import spacy
import spacy_transformers
from spacy.tokens import DocBin

TRAIN = os.path.join('/home/sakshi/projects/twt-sentiment-analysis/data/processed_data', 'train.csv')
TEST = os.path.join('/home/sakshi/projects/twt-sentiment-analysis/data/processed_data', 'test.csv')

nlp = spacy.load('en_core_web_trf')

def convert2doc(data):
    texts = []
    for doc, label in tqdm(nlp.pipe(data, as_tuples = True), total = len(data)):
        if label == 0:
            if label == 0:
                doc.cats['positive'] = 0
                doc.cats['negative'] = 1
            else:
                doc.cats['positive'] = 1
                doc.cats['negative'] = 0
            texts.append(doc)
    return texts

def load_data_and_prepare_for_training(dirpath):
    df_train = pd.read_csv(TRAIN)
    df_test = pd.read_csv(TEST)

    df_train0 = df_train.loc[df_train['sentiment'] == 0]
    df_train4 = df_train.loc[df_train['sentiment'] == 4]

    df_train0 = df_train0.sample(frac = 1).iloc[:5000]
    df_train4 = df_train4.sample(frac = 1).iloc[:5000]

    df_train = pd.concat([df_train0, df_train4])

    df_train['sentiment'] = df_train['sentiment'].replace(4, 1)
    df_test = df_test.loc[df_test['sentiment'] != 2]
    df_test['sentiment'] = df_test['sentiment'].replace(4, 1)

    train, dev = train_test_split(df_train, train_size = 0.85, random_state = 42, shuffle = True)

    train['tuple'] = train.apply(lambda x : (x['cleaned_tweet'], x['sentiment']), axis = 1)
    train_tuples = train['tuple'].to_list()

    dev['tuple'] = dev.apply(lambda x : (x['cleaned_tweet'], x['sentiment']), axis = 1)
    dev_tuples = dev['tuple'].to_list()

    test = df_test
    test['tuple'] = df_test.apply(lambda x : (x['cleaned_tweet'], x['sentiment']), axis = 1)
    test_tuples = df_test['tuple'].to_list()

    train_doc_bin = DocBin(docs = convert2doc(train_tuples))
    dev_doc_bin = DocBin(docs = convert2doc(dev_tuples))
    test_doc_bin = DocBin(docs = convert2doc(test_tuples))

    train_doc_bin.to_disk(os.path.join(dirpath, 'train.spacy'))
    dev_doc_bin.to_disk(os.path.join(dirpath, 'dev.spacy'))
    test_doc_bin.to_disk(os.path.join(dirpath, 'test.spacy'))

if __name__ == '__main__':
    dirpath = '/home/sakshi/projects/twt-sentiment-analysis/data/processed_data/spacy'
    load_data_and_prepare_for_training(dirpath)

"""
To train spacy classifier

run the following commands after running this file
1. download base_config.cfg from spacy quickstarts for textcat pipeline
2. add the train, dev, test paths in base_config.cfg
3. to generate final config, fire
   python -m spacy init fill-config base_config.cfg config.cfg
4. to train, fire
    python -m spacy train config.cfg --verbose --output /home/sakshi/projects/twt-sentiment-analysis/models/spacy-classifier
"""
