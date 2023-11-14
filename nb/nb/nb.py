import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class NaiveBayes:
    """
    contain the functions that deal with the training and 
    testing of the Naive Bayes classifier
    """
    
    train_df = []
    test_df=[]
    vocabulary=[]
    priors=[]
    likelihoods=[]
    class_predictions=[]
    accuracy=[] 
    f1_score=[] 
    conf_matrix=[]
    def __init__(self):
    #def __init__(self, indir,training_df,test_df,vocabulary,priors,likelihoods,class_predictions):
        pass

    def train_nb(self,df, alpha=0.1):
        """
        Takes as input a pandas DataFrame containing Federalist
        files text to determine priors and likelihoods
        :param df: a pandas DataFrame
        :return: two numpy arrays for the priors and likelihoods
        """
        vocabulary={}
        index_t=0
        for i in df['text']:
            tokens=i.split()
            for token in tokens:#iterate through the tokens, ignore the token has already in the voc and add new token
                if token not in vocabulary:
                    vocabulary[token]=index_t
                    index_t=index_t+1
        n_docs = df.shape[0]
        n_classes = df.shape[1]
        priors = np.array([sum(df['author'] == auth) / n_docs for auth in range(
            n_classes)])
        # Create a matrix containing all 0s called training_matrix of size (n_docs,
        # len(vocabulary)), then fill it with the counts of each word for each
        # document
        # this is the bag-of-words matrix for all the documents
        training_matrix = np.zeros(shape=(n_docs, len(vocabulary)))
        for idx, document in enumerate(df['text'].tolist()):
            for token in document.split():
                j = vocabulary[token]
                training_matrix[idx, j] += 1
        # get word counts for both classes
        word_counts_per_class = {auth: np.sum(training_matrix[np.where(
            df['author'] == auth)]) for auth in range(n_classes)}
        likelihoods = np.zeros(shape=(n_classes, len(vocabulary)))

        for token, idx in vocabulary.items():
            for auth in range(n_classes):
                count_token_idx_in_class_auth = np.sum((training_matrix[
                                            np.where(df['author'] == auth), idx]))
                likelihoods[auth, idx] = (alpha +
                    count_token_idx_in_class_auth) / \
                    (alpha * (len(vocabulary) + 1) + word_counts_per_class[auth])
        return vocabulary, priors, likelihoods


    def test(self,df, vocabulary, priors, likelihoods):
        """
        Takes as input a pandas DataFrame representing the disputed Federalist
        Papers and returns predictions for every text document
        :param df: a pandas DataFrame
        :return: a numpy array of predictions
        """
        class_predictions = []
        for text in df['text']:
            test_vector = np.zeros(shape=(len(vocabulary)))#Fill test_vector with counts for the words that appear in the vocabulary
            for token in text.split():
                # skip the words that do not appear in the training corpus
                if token in vocabulary:
                    idx = vocabulary[token]
                    test_vector[idx] += 1
            # compute predictions p(y|test)
            preds = test_vector.dot(np.log(likelihoods).T) + np.log(priors)
            yhat = np.argmax(preds)
            class_predictions.append(yhat)  
        return class_predictions