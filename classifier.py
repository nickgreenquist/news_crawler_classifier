import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import string
import math
from math import sqrt
import random
import sys
import os
import re
import collections
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

removephrases = [
    'media playback is unsupported on your device',
    'image copyright',
    'getty images',
    'media caption',
    'image caption',
    '(reuters)',
    '(upi)',
    'getty',
    '(cnn)',
    'advertisement',
]

class ArticleCrawled():
    def __init__(self, heading, text, link, signature, shingles):
        self.heading = heading
        self.text = text
        self.link = link
        self.signature = signature
        self.shingles = shingles

class Category():
    def __init__(self, name, links, token_list, articles, test, test_articles, test_categories, train):
        self.name = name
        self.links = links
        self.token_list = token_list #a token is a sentence
        self.test = test
        self.test_articles = test_articles
        self.test_categories = test_categories
        self.train = train
        self.articles = articles

categories = []
train_articles = []
train_categories = []

def ReadDataSet():
    print("Reading in data from files")
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
                    new_category = Category(name = filename.split('.')[0], links = [], token_list = [], articles = [], train = [], test = [], test_articles = [], test_categories = [])
                    categories.append(new_category)

    #read articles into category articles
    total_articles = 0
    seen = set()
    for category in categories:
        for dirname in os.listdir(data_dir):
            try:
                filename = data_dir + "/" + dirname + "/" + category.name + ".txt"
                file = open((filename),"r") 
                for line in file: 
                    #must contain letters and be longer than 5 characters long
                    if any(c.isalpha()for c in line) and len(line) > 5:
                        article = line.split(":::::")
                        heading = article[0]
                        text = article[1]

                        if line not in seen:
                            seen.add(line)

                            minArticleLength = 100
                            if len(line.split()) > minArticleLength:
                                total_articles += 1
                                newArticle = ArticleCrawled(heading = heading, text = heading + ' ' + text, link = None, signature = [], shingles = [])
                                category.articles.append(newArticle)
                file.close()
            except Exception as e:
                print(e)
    print ("Total Articles read: " + str(total_articles))


#trim out dataset
def CleanData():
    print("Cleaning data")
    cachedStopWords = stopwords.words("english")
    for category in categories:
        newList = []
        i = 0
        while i < len(category.articles):
            category.articles[i].text = category.articles[i].text.lower()

            #remove header divider
            category.articles[i].text = re.sub(":::::", ' ', category.articles[i].text)   

            #remove bad phrases
            for phrase in removephrases:
                category.articles[i].text = re.sub(phrase, '', category.articles[i].text)   

            #remove easy punctation
            category.articles[i].text = re.sub(r"[,.;@#?!&$-]+\ *", " ", category.articles[i].text)   

            #remove punctuation
            category.articles[i].text = "".join(l for l in category.articles[i].text if l not in string.punctuation)   

            #remove stop words
            category.articles[i].text = ' '.join([word for word in category.articles[i].text.split() if word not in cachedStopWords])

            #finally condense whitespace
            category.articles[i].text = re.sub('\s+',' ',category.articles[i].text)

            i += 1
    print("Finished cleaning data")

def PrepareTestAndTrain():
    #shuffle arrays
    for category in categories:
        random.shuffle(category.articles)

    #find train and test sets for each category
    for category in categories:
        c_length = len(category.articles)
        category.train = category.articles[: int(.8 *c_length)]
        category.test = category.articles[int(.8 *c_length):c_length]
        for article in category.test:
            category.test_articles.append(article.text)
            category.test_categories.append([category.name])

    #extract train articles into array
    for category in categories:
        for article in category.train:
            train_articles.append(article.text)
            train_categories.append([category.name])

'''---------------------------------------------OneVsOneClassifier--------------------------------------'''
def OneVsOne():
    print ("Results for OneVsOneClassifier")
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsOneClassifier(LinearSVC()))])
    Y_train = np.array(train_categories)
    X_train = np.array(train_articles)
    classifier.fit(X_train, Y_train.ravel())

    for category in categories:
        X_test = np.array(category.test_articles)
        predicted = classifier.predict(X_test)
        print (category.name + ": " + str(np.mean(predicted == np.array(category.test_categories))))

        #see why business has bad results
        '''if "politics" in category.name.lower():
            for i in range(0, len(predicted)):
                if "politics" not in predicted[i].lower():
                    print("%s: %s" % (predicted[i], category.test[i].heading))'''
    print('\n')

'''-------------------------------------------Naives Bayes----------------------------------------------------'''
def NaiveBayes():
    print ("Results for Naive Bayes")
    Y_train = np.array(train_categories)
    X_train = np.array(train_articles)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    classifier = MultinomialNB().fit(X_train_tfidf, Y_train.ravel())

    for category in categories:
        X_new_counts = count_vect.transform(category.test_articles)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        predicted = classifier.predict(X_new_tfidf)
        print (category.name + ": " + str(np.mean(predicted == np.array(category.test_categories))))
    print ('\n')

'''-------------------------------------------Support Vector Machine----------------------------------------------------'''
def SVM():
    print ("Results for Support Vector Machine")
    classifier = Pipeline([('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
            alpha=1e-3, random_state=42,
            max_iter=5, tol=None)),
    ])
    Y_train = np.array(train_categories)
    X_train = np.array(train_articles)
    classifier.fit(X_train, Y_train.ravel())  

    for category in categories:
        predicted = classifier.predict(category.test_articles)
        print (category.name + ": " + str(np.mean(predicted == np.array(category.test_categories))))
    print ('\n')

def SupervisedLearning():
    PrepareTestAndTrain()
    OneVsOne()
    NaiveBayes()
    SVM()

def KmeansSKL():
    articles = []
    for category in categories:
        for article in category.articles:
            articles.append(article.text)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(articles)
    true_k = 7
    model = KMeans(n_clusters=6, random_state=0, max_iter=100).fit(X)

    i = 0
    for category in categories:
        start = i
        for article in category.articles:
            i += 1
        end = i
        print("%s: %s" % (category.name, model.labels_[start:end]))

'''-------------------------------------------------------------------driver---------------------------------------------------------'''
ReadDataSet()
CleanData()

SupervisedLearning()
KmeansSKL()