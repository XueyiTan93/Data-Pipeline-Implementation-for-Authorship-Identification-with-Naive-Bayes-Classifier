# What this code does
This code writes a Naive Bayes classifier with Lidstone smoothing(default alpha is 0.1) to perform authorship identification over some speeches txt files.

The code took these following steps:

1.Package utils stores data preprocessing method, function extract_text. The purpose is to build training and testing dataframe to feed train_nb method in NaiveBayes class

2.Write NaiveBayes class in nb.py

3.unit tests. Put small testing data in tests/data/docs.txt. In test_naive_bayes.py, from nb.nb import NaiveBayes, call the class method, get the priors, likelihood, and test whether they are correct by defining test function (line41 till the end in test_naive_bayes.py)

4.then install nb.py as package
(How to install a mini package)
We install this package by running `pip install -e .` at the top-level of the folder structure (i.e., at the same level where `setup.py` sits).

5.pytest. Run "pytest tests/" in the terminal at folder 'nb' (assigment3\nb)
For example:
PS C:\Users\lenovo\OneDrive\Desktop\5400nlp\assignment3\nb> pytest tests/

6.Put everythin together in main.py using argparse(see **how to use**)
# How to use
running in the terminal using this command:
python ./nb/bin/main.py -f ./data
run main.py and provide data folder, using relative path to the current working terminal
for example:
PS C:\Users\lenovo\OneDrive\Desktop\5400nlp\assignment3\nb> python ./nb/bin/main.py -f ./data

# Reference
For the training and testing method code, I referenced Professor Trevor Adriaanse's code in my own work.
Things my code for assignment 2 should modify: 
1.In my testing method, I didn't specify "if token in vocabulary:", which skipped words that do not appear in the training vocabulary, resulting in wrong classfication.
2.Comparing to mine, professor made "vocabulary" of training data a dictionary not a list, which is more intuitive and convenient.

Things Professor's code should have done better:
1.In function train_nb(), it seems more straighforward to use enumerate, but every time running it may result a different order of word in vocabulary, which cause trouble to pytest. Therefore, I modified this part using a for loop beginning from the first row in df['text'] and now the order of word in vocabulary is stable.

My code for creating vocabulary:
vocabulary={}
    index_t=0
    for i in df['text']:
        tokens=i.split()
        for token in tokens:#iterate through the tokens, ignore the token has already in the voc and add new token
            if token not in vocabulary:
                vocabulary[token]=index_t
                index_t=index_t+1