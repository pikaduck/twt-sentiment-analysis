"""
    @author : Sakshi Tantak
"""

# Imports
import pandas as pd

from datasets import Dataset

from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

def prepare_dataset(per_class_sample_size, train_data_path, test_data_path):
    """
        Function to prepare dataset for training
        Args:
            @param per_class_sample_size (int) : No. of samples to take in training data for each class
            @param train_data_path (str or Path) : Path to training data csv (train.csv)
            @param test_data_path (str or Path) : Path to testing data csv (test.csv)
        Return:
            @return train_dataset (datasets.Dataset) : Train Dataset
            @return test_dataset (datasets.Dataset) : Test Dataset
    """
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    df_train = pd.concat([
        df_train.loc[df_train['sentiment'] == 0].sample(frac = 1).iloc[:per_class_sample_size],
        df_train.loc[df_train['sentiment'] == 4].sample(frac = 1).iloc[:per_class_sample_size]
    ])

    df_train['sentiment'] = df_train['sentiment'].replace(4, 1)
    df_test = df_test.loc[df_test['sentiment'] != 2]
    df_test['sentiment'] = df_test['sentiment'].replace(4, 1)

    df_train['label_text'] = df_train['sentiment'].apply(lambda x : 'positive' if x==1 else 'negative')
    df_test['label_text'] = df_test['sentiment'].apply(lambda x : 'positive' if x==1 else 'negative')

    df_train = df_train.rename(columns = {'sentiment' : 'label', 'cleaned_tweet' : 'text'})
    df_test = df_test.rename(columns = {'sentiment' : 'label', 'cleaned_tweet' : 'text'})

    df_train = df_train[['text', 'label', 'label_text']]
    df_test = df_test[['text', 'label', 'label_text']]

    train_dataset = Dataset.from_pandas(df_train, split = 'train').remove_columns('__index_level_0__')
    test_dataset = Dataset.from_pandas(df_test, split = 'test').remove_columns('__index_level_0__')

    return train_dataset, test_dataset

class ClassificationTrainer:
    """
        Text-classification trainer class
    """
    def __init__(self, model_name_or_path):
        """
            Args:
                @param model_name_or_path (str or Path) : Name of or path to the base sentence transformer model to be used
        """
        self.model = SetFitModel.from_pretrained(model_name_or_path)

    def train(self, train_ds, test_ds, batch_size, num_iterations, num_train_epochs):
        """
            Function to fine-tune the sentence transformers
            Args:
                @param train_ds (datasets.Dataset) : Train dataset
                @param test_ds (datasets.Dataset) : Test dataset
                @param batch_size (int) : Per-device train & test batch size
                @param num_iterations (int) : No. of text pairs to generate for contrastive learning
                @param num_train_epochs (int) : No. of train epochs
        """
        self.trainer = SetFitTrainer(
            model = self.model,
            train_dataset = train_ds,
            eval_dataset = test_ds,
            loss_class = CosineSimilarityLoss,
            batch_size = batch_size,
            num_iterations = num_iterations,
            num_epochs = num_train_epochs
        )

        self.trainer.train()

    def evaluate(self):
        """
            Function to evaluate model on test data
        """
        return self.trainer.evaluate()

    def save_model(self, save_model_dirpath):
        """
            Function to save fine-tuned model
            Args:
                @param save_model_dirpath (str or Path) : Path to directory to save model in
        """
        self.model.save_pretrained(save_model_dirpath)
        print(f'Model saved at {save_model_dirpath}')

    def predict(self, text):
        """
            Function to test fine-tuned model
            Args:
                @param text (str) : Text
            Returns:
                @param label (str) : 'positive' or 'negative' label
        """
        return 'positive' if self.model([text]).item()==1 else 'negative'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_path', type = str,
                        default = '/home/sakshi/projects/twt-sentiment-analysis/data/processed_data/train.csv')
    
    parser.add_argument('--test_data_path', type = str,
                    default = '/home/sakshi/projects/twt-sentiment-analysis/data/processed_data/test.csv')
    
    parser.add_argument('--model_name_or_path', type = str, default = 'sentence-transformers/paraphrase-mpnet-base-v2')
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_iterations', type = int, default = 20)
    parser.add_argument('--num_train_epochs', type = int, default = 1)
    parser.add_argument('--per_class_sample_size', type = int, default = 64)
    parser.add_argument('--save_model_dirpath', type = str, default = '/home/sakshi/projects/twt-sentiment-analysis/models/setfit-classifier')

    args = parser.parse_args()

    train_ds, test_ds = prepare_dataset(args.per_class_sample_size)
    classification_trainer = ClassificationTrainer(args.model_name_or_path)
    classification_trainer.train(train_ds, test_ds, args.batch_size, args.num_iterations, args.num_train_epochs)
    metrics = classification_trainer.evaluate()
    classification_trainer.save_model(args.save_model_dirpath)
    print(f'Metrics : {metrics}')
