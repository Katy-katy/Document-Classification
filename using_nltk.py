import sys, random, nltk
from nltk import bigrams

selected_features = None

stopwords = ['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 't', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 's', 'or', 'own', 'into', 'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'yours', 'so', 'the', 'having', 'once']

def add_lexical_features(fdist, feature_vector, text):
    feature_vector["len"] = len(text)
    text_nl = nltk.Text(text)
    for word, freq in fdist.items():
        fname = "UNI_" + word 
        if selected_features == None or fname in selected_features:        
            #feature_vector[fname] = text_nl.count(word)
            feature_vector[fname] = 1
            
def features(review_words):
    feature_vector = {}

    uni_dist = nltk.FreqDist(review_words)
    my_bigrams = list(bigrams(review_words))
    bi_dist = nltk.FreqDist(my_bigrams)
    
    add_lexical_features(uni_dist,feature_vector, review_words)
    
    return feature_vector

#to train the model
#https://s3.amazonaws.com/hr-testcases/597/assets/trainingdata.txt
file = open("trainingdata.txt", "rb")
raw_data = file.read().decode("latin1")
file.close()

docs = raw_data.split("\n")
docs2 = docs[1: ]
train = []
for d in docs2:
    d = d.split()
    if len(d)!=0:
        cl = d[0]
        text_d = d[1: ]#we need to remove the stopwords
        text = []
        for w in text_d:
            if w not in stopwords:
                text.append(w)
        item = (text, cl)
        train.append(item)

random.seed(0)
random.shuffle(train)

train_set = train[ :4485]
valid_set = train[4486: ]

featuresets_tr = [(features(words), label) for (words, label) in train_set ]
featuresets_val = [(features(words), label) for (words, label) in valid_set ]

#featuresets = [(features(words), label) for (words, label) in train ]

classifier = nltk.NaiveBayesClassifier.train(featuresets_tr)

#classifier = nltk.NaiveBayesClassifier.train(featuresets)

#classifier.show_most_informative_features(50)
accuracy = nltk.classify.accuracy(classifier, featuresets_val)
#print(accuracy)




#to take input
n = int(input().strip())
a = []
for a_i in range(n): # to read a matrix
    a_t = [a_temp for a_temp in input().strip().split(' ')]
    a.append(a_t)
    
featuresets_test = [features(words) for words in a ]  

predicted_labels = classifier.classify_many(featuresets_test)
for l in predicted_labels:
    print (int(l))