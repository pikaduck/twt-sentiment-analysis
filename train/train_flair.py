"""
    @author : Sakshi Tantak
"""

# Imports
import os
import pathlib
import pandas as pd

from sklearn.model_selection import train_test_split

from flair.datasets import CSVClassificationCorpus
from flair.embeddings import (DocumentLSTMEmbeddings, FlairEmbeddings,
                              DocumentPoolEmbeddings)
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


TRAIN = os.path.join('/home/sakshi/projects/twt-sentiment-analysis/data/processed_data', 'train.csv')
TEST = os.path.join('/home/sakshi/projects/twt-sentiment-analysis/data/processed_data', 'test.csv')

def load_data_and_prepare_for_training(dirpath):
    df_train = pd.read_csv(TRAIN)
    df_test = pd.read_csv(TEST)

    df_train0 = df_train.loc[df_train['sentiment'] == 0]
    df_train4 = df_train.loc[df_train['sentiment'] == 4]

    df_train0 = df_train0.sample(frac = 1).iloc[:5000]
    df_train4 = df_train4.sample(frac = 1).iloc[:5000]

    df_train['sentiment'] = df_train['sentiment'].replace(4, 1)
    df_test = df_test.loc[df_test['sentiment'] != 2]
    df_test['sentiment'] = df_test['sentiment'].replace(4, 1)

    df_train['sentiment'] = df_train['sentiment'].astype(str)
    df_test['sentiment'] = df_test['sentiment'].astype(str)

    train, dev = train_test_split(df_train, train_size = 0.85, random_state = 42, shuffle = True)

    train = train.rename(columns = {'cleaned_tweet' : 'text', 'sentiment' : 'labels'})
    train = train[['labels', 'text']]

    dev = dev.rename(columns = {'cleaned_tweet' : 'text', 'sentiment' : 'labels'})
    dev = dev[['labels', 'text']]

    test = df_test.rename(columns = {'cleaned_tweet' : 'text', 'sentiment' : 'labels'})
    test = test[['labels', 'text']]

    train.to_csv(os.path.join(dirpath, 'train_corpus.csv'), index = False, sep = '\t')
    dev.to_csv(os.path.join(dirpath, 'dev_corpus.csv'), index = False, sep = '\t')
    test.to_csv(os.path.join(dirpath, 'test_corpus.csv'), index = False, sep = '\t')

class ClassificationTrainer:
    def __init__(self, data_dirpath):
        self.column_name_map = {0 : 'labels', 1 : 'text'}
        self.label_type = 'labels'

        self.corpus = CSVClassificationCorpus(
            data_dirpath, column_name_map = self.column_name_map, skip_header = True,
            delimiter = '\t', train_file = 'train_corpus.csv', dev_file = 'dev_corpus.csv',
            test_file = 'test_corpus.csv'
        )
        self.label_dict = self.corpus.make_label_dictionary(label_type = self.label_type)

    def train(self, save_model_path):
        word_embeddings = [FlairEmbeddings('news-forward-fast')]

        document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size = 512, reproject_words = True,
                                            reproject_words_dimension=256)
        classifier = TextClassifier(document_embeddings, label_dictionary = self.label_dict, label_type = self.label_type)

        trainer = ModelTrainer(classifier, self.corpus)

        trainer.train(save_model_path, learning_rate = 0.01, mini_batch_size = 64, max_epochs = 35,
                        embeddings_storage_mode = 'gpu',
                    )

if __name__ == '__main__':
    dirpath = '/home/sakshi/projects/twt-sentiment-analysis/data/processed_data/flair'

    load_data_and_prepare_for_training(dirpath)

    classification_trainer = ClassificationTrainer(dirpath)
    classification_trainer.train('/home/sakshi/projects/twt-sentiment-analysis/models/flair-classifier')