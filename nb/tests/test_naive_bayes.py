from nb.nb import NaiveBayes
import numpy as np
import pandas as pd
from pathlib import Path
#input data from tests.data.docs.txt


# Create lists to hold the classes and texts
classes = []
texts = []

# Read the file line by line
with open("./tests/data/docs.txt") as fp:
    lines = fp.readlines()

    for line in lines:
        # Split the line into class and text parts
        class_part, text_part = line.strip().split(':')

        # Extract the class information
        class_info = class_part.split('(')[-1].split(')')[0]

        # Append the class and text information to the lists
        classes.append(class_info)
        texts.append(text_part)

# Create a DataFrame from the lists
training_df = pd.DataFrame({'author': classes, 'text': texts})
author_to_id_map = {'class 1': 0, 'class 2': 1}
# replace the strings for the author names with numeric codes (0, 1)
training_df['author'] = training_df['author'].apply(lambda x:author_to_id_map.get(x))

mynb = NaiveBayes()
vocabulary, priors, likelihoods = mynb.train_nb(training_df)
class_predictions = mynb.test(training_df, vocabulary, priors, likelihoods)

print(priors)
print(vocabulary)
print(likelihoods)
print(class_predictions)
def test_priors():
    #correct_priors=np.array([0.75,0.25])
    assert priors[0] == 0.75
    assert priors[1] ==0.25
def test_likelihood():
    correct_likelihood=np.array([[0.5862069 , 0.12643678, 0.12643678, 0.12643678, 0.01149425,
        0.01149425],
       [0.2972973 , 0.02702703, 0.02702703, 0.02702703, 0.2972973 ,
        0.2972973 ]])
    are_arrays_equal = np.allclose(likelihoods, correct_likelihood)
    assert are_arrays_equal == True, "likelihoods is not correct!"
def test_sum():
    assert priors.sum() ==1,"The sum of priors is not 1"