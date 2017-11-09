import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
import string
import random
import sys
import os
import re
import collections
from nltk.stem.wordnet import WordNetLemmatizer
from stemming.porter2 import stem
lmtzr = WordNetLemmatizer()

class Category():
    def __init__(self, name, links, token_list, articles, test, train):
        self.name = name
        self.links = links
        self.token_list = token_list #a token is a sentence
        self.test = test
        self.train = train
        self.articles = articles

categories = []
x_train_articles = []
y_train_tags = []

def ReadDataSet():
    total_sentences = 0
    data_dir = os.getcwd() + '/data'
    if len(sys.argv) > 1:
        data_dir = os.getcwd() + '/' + sys.argv[1]
    for dirname in os.listdir(data_dir):
        for filename in os.listdir(data_dir + "/" + dirname):
            if 'articles' in filename:
                #create new category if needed
                category_exists = False
                for category in categories:
                    if category.name == filename.split('.')[0]:
                        category_exists = True
                if not category_exists:
                    new_category = Category(name = filename.split('.')[0], links = [], token_list = [], articles = [], train = [], test = [])
                    categories.append(new_category)

    #read articles into category articles
    total_articles = 0
    for category in categories:
        for dirname in os.listdir(data_dir):
            file = open((data_dir + "/" + dirname + "/" + category.name + ".txt"),"r") 
            for line in file: 
                #must contain letters and be longer than 5 characters long
                if any(c.isalpha()for c in line) and len(line) > 5:
                    total_articles += 1

                    article = line.split(":::::")
                    #print(article[0])

                    category.articles.append(line)
            file.close()
    print ("Total Articles: " + str(total_articles))


#trim out dataset
def CleanData():
    #remove duplicate articles
    total_articles = 0
    for category in categories:
        withDupes = len(category.articles)
        category.articles = list(set(category.articles))
        removed = withDupes - len(category.articles)

        print("%s # of dupes: %s" % (category.name, removed))

        total_articles += len(category.articles)
    print ("Total Articles with removed duplicates: " + str(total_articles))
    print ('\n')

    cachedStopWords = stopwords.words("english")
    for category in categories:
        newList = []
        i = 0
        while i < len(category.articles):
            heading = category.articles[i].split(":::::")[0]

            category.articles[i] = category.articles[i].lower()

            #remove easy punctation
            category.articles[i] = re.sub(r"[,.;@#?!&$-]+\ *", " ", category.articles[i])   

            #remove punctuation
            category.articles[i] = "".join(l for l in category.articles[i] if l not in string.punctuation)   

            category.articles[i] = re.sub('\s+',' ',category.articles[i])

            #remove stop words
            text = ' '.join([word for word in category.articles[i].split() if word not in cachedStopWords])

            text = heading + " ::::: " + text

            category.articles[i] = text

            i += 1

def PrepareTestAndTrain():
    #shuffle arrays
    for category in categories:
        random.shuffle(category.articles)

    #find train and test sets for each category
    for category in categories:
        c_length = len(category.articles)
        category.train = category.articles[: int(.8 *c_length)]
        category.test = category.articles[int(.8 *c_length):c_length]

    #Find train articles from 80% of the articles per category
    for category in categories:
        for tok in category.train:
            x_train_articles.append(tok)
            y_train_tags.append([category.name])
    X_train = np.array(x_train_articles)
    return X_train

'''---------------------------------------------OneVsOneClassifier--------------------------------------'''
def OneVsOne(X_train):
    print ("Results for OneVsOneClassifier")
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsOneClassifier(LinearSVC()))])
    Y = np.array(y_train_tags)
    classifier.fit(X_train, Y.ravel())

    #Find test articles from 20% of the articles list per category
    target_names = []
    for category in categories:
        try:
            #grab test articles for this category
            target_names.append(category.name)
            c_length = len(category.articles)
            test_articles = []
            Y_test = []
            for tok in category.test:
                test_articles.append(tok)
                Y_test.append([category.name])

            #classify test articles   
            X_test = np.array(test_articles)
            predicted = classifier.predict(X_test)

            #results
            print (category.name + ": " + str(np.mean(predicted == np.array(Y_test))))

            if "business" in category.name.lower():
                for i in range(0, len(predicted)):
                    if "business" not in predicted[i].lower():
                        print("%s: %s" % (predicted[i], X_test[i].split(":::::")[0]))

        except:
            print(category.name + ": ERROR")
    print('\n')

    '''s = ""
    while s != 'q':
        s = input()
        r = classifier.predict([s])
        print("%s: %s" % (s, r))'''

'''-------------------------------------------Naives Bayes----------------------------------------------------'''
def NaiveBayes(X_train):
    print ("Results for Naive Bayes")
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)

    from sklearn.feature_extraction.text import TfidfTransformer
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    from sklearn.naive_bayes import MultinomialNB
    Y = np.array(y_train_tags)
    clf = MultinomialNB().fit(X_train_tfidf, Y.ravel())

    for category in categories:
        c_length = len(category.articles)
        test_articles = []
        Y_test = []
        for tok in category.test:
                test_articles.append(tok)
                Y_test.append([category.name])


        X_new_counts = count_vect.transform(test_articles)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        predicted = clf.predict(X_new_tfidf)

        #results
        print (category.name + ": " + str(np.mean(predicted == np.array(Y_test))))
    print ('\n')

'''-------------------------------------------Support Vector Machine----------------------------------------------------'''
def SVM(X_train):
    print ("Results for Support Vector Machine")
    from sklearn.linear_model import SGDClassifier
    text_clf = Pipeline([('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
            alpha=1e-3, random_state=42,
            max_iter=5, tol=None)),
    ])
    Y = np.array(y_train_tags)
    text_clf.fit(X_train, Y.ravel())  

    for category in categories:
        c_length = len(category.articles)
        test_articles = []
        Y_test = []
        for tok in category.test:
                test_articles.append(tok)
                Y_test.append([category.name])

        predicted = text_clf.predict(test_articles)
        
        #results
        print (category.name + ": " + str(np.mean(predicted == np.array(Y_test))))

def SupervisedLearning():
    X_train = PrepareTestAndTrain()
    OneVsOne(X_train)
    NaiveBayes(X_train)
    SVM(X_train)

ReadDataSet()
CleanData()

#SupervisedLearning()




'''-------------------------------------------------------------------'''



#Unsupervised Learning
numHashes = 100
nextPrime = 4294967311
maxShingleID = 2**32-1
iterations = 30

def pickRandomCoeffs(k):
  # Create a list of 'k' random values.
  randList = []
  
  while k > 0:
    # Get a random shingle ID.
    randIndex = random.randint(0, maxShingleID) 
  
    # Ensure that each random number is unique.
    while randIndex in randList:
      randIndex = random.randint(0, maxShingleID) 
    
    # Add the random number to the list.
    randList.append(randIndex)
    k = k - 1
    
  return randList

coeffA = pickRandomCoeffs(numHashes)
coeffB = pickRandomCoeffs(numHashes)
print("random done")

def characteristicMatrix():
    kvp = []
    for category in categories:
        for article in category.articles:
            split = article.split(":::::")
            heading = split[0]
            text = split[1]
            text = text.split()
            text = list(set(text))
            hashlist = []
            for word in text:
                crc = hash(word)
                hashlist.append(crc)
            kvp.append((heading, hashlist))
    return kvp

def minHash(kvp):
    signatures = []

    print(len(kvp))
    for doc in kvp:
        shingleIDSet = doc[1]
        signature = []
        
        for i in range(0, numHashes):
            minHashCode = nextPrime + 1
            
            for shingleID in shingleIDSet:
                hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime 
                hashCode = hashCode / nextPrime
                
                if hashCode < minHashCode:
                    minHashCode = hashCode

            signature.append(minHashCode)    

        signatures.append((doc[0], signature))  

    return signatures

def jacdist(x, c):
    same = 0
    total = 0
    unique = []

    for i in range(0, numHashes):
        if x[i] == c[i]:
            same += 1
        unique.append(x[i])
        unique.append(c[i])
    
    unique = set(unique)
    unique = list(unique)
    total = len(unique)
    dist = 1.0 - (float(same) / float(total))
    print(dist)
    return dist


kvp = characteristicMatrix()
signatures = minHash(kvp)

print(signatures[::10])
