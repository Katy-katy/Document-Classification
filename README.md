# Document-Classification
Document Classification using Python and scikit-learn.

Thus project was written as an programming exercise in text classification:

https://www.hackerrank.com/challenges/document-classification


My task is to classify given documents into one of eight categories: [1,2,3,…8].

### Training Data

The "trainingdata.txt" file will be included with my program at runtime.

The file is formatted as follows:

The first line contains the number of lines that will follow.

Each following line will contain a number (1-8), which is the category number. The number will be followed by a space then some space seperated words which is the processed document.

### Input (Test)

The first line in the input file will contain T the number of documents. T lines will follow each containing a series of space seperated words which represents the processed document. For example:

3 

This is a document 

this is another document 

documents are seperated by newlines


### Output

For each document output a number between 1-8 which you believe this document should be categorized as. For example:

1 

4 

8

## My Work

I tried to use NLTK with removing stop words and using bigrams as features. I tried some classifiers. The best result was given by NaiveBayesClassifier. I got about 75 % accuracy on the test set.

But then I realized that using feature extractor TfidfTransformer from scikit-learn with some scikit-learn classifiers algorithms gave much better result. I tried some different algorithms and finally got about 97% accuracy using  VotingClassifier with RandomForestClassifier and KNeighborsClassifier as estimators.


