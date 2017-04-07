import sys, zlib, pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.decomposition import TruncatedSVD

from sklearn.neighbors import KNeighborsClassifier

def stopw(w): 
    stopwords = ['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 't', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 's', 'or', 'own', 'into', 'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'yours', 'so', 'the', 'having', 'once']
    return w in stopwords

#to train the model
#https://s3.amazonaws.com/hr-testcases/597/assets/trainingdata.txt
file = open("trainingdata.txt", "rb")
raw_data = file.read().decode("latin1")
file.close()

docs = raw_data.split("\n")
docs2 = docs[1: ]
train = []
labels = []
for d in docs2:
    d = d.split()
    if len(d)!=0:
        cl = d[0]
        labels.append(int(cl))
        text_d = d[1: ]#we need to remove the stopwords
        train.append(' '.join([i for i in d[1:] if not stopw(i)]))
        

def work(test_data):
    my_vector = CountVectorizer(input='content',ngram_range=(1,2))
    X_train_counts = my_vector.fit_transform(train,)
    tf_transformer = TfidfTransformer(use_idf=True,).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    svd = TruncatedSVD(n_components=50, random_state=9)
    X_train = svd.fit_transform(X_train_tf)
    
    clf = KNeighborsClassifier(n_neighbors=8).fit(X_train, labels)
    
    X_new_counts = my_vector.transform(test_data)
    X_new_tfidf = tf_transformer.transform(X_new_counts)
    X_new = svd.transform(X_new_tfidf)
    return clf.predict(X_new)

n = int(input())
a = []
for a_i in range(n):
    a_t = input()
    a.append(' '.join([i for i in a_t.split() if not stopw(i)]))
        
predicted_labels = work(a)
for l in predicted_labels:
    print (l)