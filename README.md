# Twitter Data Sentiment Analysis

## Text-preprocessing approach undertaken for ML models
1. text changed to lowercase & twitter username mentions removed
2. Basic cleaning wherein any words like nnoooo, nope, etc were changed to no
3. !!! was changed to !, similarly ... was changed to .
4. abbrevations/contractions expanded, ex. couldn't --> could not. can't --> can not and so on
5. websites & urls cleaned
6. any emojis replaced with their text
7. Exclusive to ML models - punctuations removed, numbers removed, stopwords removed & text lemmatized
8. Deduplication done considering - sentiment, tweet, username

## Data subsetting
1. Data was split into to parts, class0 and class4
2. Lengths of tweets were taken into consideration to further split them into 2 parts - tweets below median length & tweets above median length
3. In each of the 4 parts, deduplication was done using username of the tweet to prevent over-representation of any tweeting/writing styles/patterns in the subset chosen for training
4. 500 samples from each part were chosen, so 500 samples of class0 with length > median & 500 with length < median. Similarly 500 samples of class4 with length > median & 500 samples with length < median
5. label 4 was replaced with 1 to aid binary classification
5. Bag-of-words generated using CountVectorizer and tfIdf used to generate vectors for training

## Models
1. ML Models were trained using Bag-of-words & TFIDF
2. SpaCy model was trained using pretrained transformers RoBerta model
3. Flair text classifier was trained
4. SetFit (sentence transformers) were trained

## Repository structure
1. Any notebooks can be found in /notebooks
2. Training scripts can be found in /train
3. Inference scripts can be found in /scripts
4. /processed_data/ has datasets processed & used for individual models, like flair, spacy
5. /trainingandtestingdata/ has data given initially