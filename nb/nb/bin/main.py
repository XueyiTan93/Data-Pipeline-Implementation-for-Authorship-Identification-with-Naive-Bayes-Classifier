from nb.nb import NaiveBayes
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from nb.utils.load_data import extract_text
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to make corpus "
                                                    "for Naive Bayes homework")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()
    
    training_df, test_df = extract_text(args.indir)
    mynb = NaiveBayes()
    vocabulary, priors, likelihoods = mynb.train_nb(training_df)
    class_predictions = mynb.test(test_df, vocabulary, priors, likelihoods)
    print(test_df)
    print(class_predictions)#kennedy': 0, 'johnson': 1, the results are all correct
    print(priors)
    #print(vocabulary)
    print(likelihoods)




