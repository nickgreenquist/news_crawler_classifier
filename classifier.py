import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
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
from stemming.porter2 import stem
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
                file = open((data_dir + "/" + dirname + "/" + category.name + ".txt"),"r") 
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
                print("ERROR")
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




'''-------------------------------------------------------Unsupervised Learning---------------------------------------------------'''

numHashes = 100
nextPrime = 4294967311
maxShingleID = 2**32-1
iterations = 30

def pickRandomCoeffs(k):
  randList = []
  while k > 0:
    randIndex = random.randint(0, maxShingleID) 
    while randIndex in randList:
      randIndex = random.randint(0, maxShingleID)  
    randList.append(randIndex)
    k = k - 1   
  return randList

def getShingles():
    print("Generating shingle set for each article")

    for category in categories:
        for article in category.articles:
            text = article.text
            heading = article.heading
            text = text.split()
            text = list(set(text))
            hashlist = []
            for word in text:
                crc = hash(word)
                hashlist.append(crc)
            article.shingles = hashlist
    print("Finished generating shingle sets")

def minHash(coeffA, coeffB):
    print("Generating signatures using minhash")
    for category in categories:
        for article in category.articles:
            shingleIDSet = article.shingles          
            for i in range(0, numHashes):
                minHashCode = nextPrime + 1           
                for shingleID in shingleIDSet:
                    hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime               
                    if hashCode < minHashCode:
                        minHashCode = hashCode
                article.signature.append(minHashCode)    
    print("Finished with minhash")

def jaccardSim(x, c):
    same = 0
    for i in range(0, numHashes):
        if x[i] == c[i]:
            same += 1   
    return  (same * 1.0) / (numHashes * 1.0)

def dist(x, c):
    sqdist = 0.
    i = 0
    for v in x:
        sqdist += (v - c[i]) ** 2
        i += 1
    return sqrt(sqdist)

def calcmean(clusterMembers):
    c = [0.] * numHashes
    n = 0
    for sig in clusterMembers:
        i = 0
        for v in sig[1]:
            c[i] += v
            i += 1
        n += 1
    for i in range(0, numHashes):
        if n < 1:
            c[i] = c[i]
        else:
            c[i] /= n
    return c

def kmeans(signatures, k):
    # Set random vectors as the centers
    centers = []

    '''centersFromSig = random.sample(signatures, k)
    for i in range(k):
        centers.append(centersFromSig[i][1])'''

    #set a center from a random article in each category
    for i in range(0, k):
        index = i % len(categories)
        randomcenter = random.sample(categories[index].articles, 1)
        centers.append(randomcenter[0].signature)

    cluster = [None] * len(signatures)

    for _ in range(0, iterations):
        i = 0
        for sig in signatures:
            cluster[i] = (sig[0], min(range(k), key=lambda j: dist(sig[1], centers[j])))
            i += 1
        j = 0
        for c in enumerate(centers):
            clusterMembers = (x for i, x in enumerate(signatures) if cluster[i][1] == j)
            temp = calcmean(clusterMembers)
            centers[j] = temp
            j += 1
    return cluster, centers

def calculateOptimalK(signatures):
    for k in range(1, 10 + 1):
        kmeansReturn  = kmeans(signatures, k)
        centers = kmeansReturn[1]
        cluster_ind = kmeansReturn[0]

        clusters = []
        for i in range(k):
            clusters.append([])
        for c in cluster_ind:
            clusters[c[1]].append(c[0])

        signaturesWithClusters = {}
        for i in range(k):
            signaturesWithClusters[i] = []

        sse = 0
        for i in range(k):
            #mean = mean(centr)
            for doc in clusters[i]:
                #print("%s,%s" % (i, doc))
                datapoints = []
                for s in signatures:
                    if s[0] == doc:
                        datapoints = s[1]
                
                clusterNum = i
                sig = datapoints
                heading = doc
                signaturesWithClusters[i].append( (clusterNum, heading, sig) )
                
                y = 0
                for datapoint in datapoints:  
                    mean = centers[i][y]
                    sse += math.pow(datapoint - mean, 2)
                    y += 1
        print(str(sse) + ',')


def UnsupervisedLearning():
    #generate random coefficients for minhash functions
    coeffA = pickRandomCoeffs(numHashes)
    coeffB = pickRandomCoeffs(numHashes)

    #represent documents as sets of shingles and then minhash them
    getShingles()
    minHash(coeffA, coeffB)

    signatures = []
    for category in categories:
        for article in category.articles:
            kvp = (article.heading, article.signature)
            signatures.append(kvp)
    
    #calculateOptimalK(signatures)

    #using elbow plots, 6 is best number of clusters
    k = 6
    kmeansReturn  = kmeans(signatures, k)
    centers = kmeansReturn[1]
    cluster_ind = kmeansReturn[0]

    clusters = []
    for i in range(k):
        clusters.append([])
    for c in cluster_ind:
        clusters[c[1]].append(c[0])

    file = open("results/cluster.txt", "w")
    i = 0
    for c in cluster_ind:
        try:
            file.write("%s,%s" % (cluster_ind[i][1], cluster_ind[i][0]))
            file.write('\n')
        except:
            file.write("ERROR")
        i += 1
    file.close()

    signaturesWithClusters = {}
    for i in range(k):
        signaturesWithClusters[i] = []

    sse = 0
    for i in range(k):
        #mean = mean(centr)
        for doc in clusters[i]:
            #print("%s,%s" % (i, doc))
            datapoints = []
            for s in signatures:
                if s[0] == doc:
                    datapoints = s[1]
            
            clusterNum = i
            sig = datapoints
            heading = doc
            signaturesWithClusters[i].append( (clusterNum, heading, sig) )
            
            y = 0
            for datapoint in datapoints:  
                mean = centers[i][y]
                sse += math.pow(datapoint - mean, 2)
                y += 1
    print(str(sse) + ',')


    '''COMPUTE JACCARD DISTANCES'''
    count = 30
    for i in range(k):
        signaturesWithClusters[i] = random.sample(signaturesWithClusters[i], count)

    file = open("results/jaccard.txt", "w")
    averages = []
    for z in range(k):
        for y in range(z, k):
            print("z:%s, y:%s" % (z,y))
            averageScore = 0
            totalScores = 0
            x = 0
            for j in range(0, len(signaturesWithClusters[z])):
                r = 0
                if z == y: #same cluster vs itself
                    r = j + 1
                for n in range(r, len(signaturesWithClusters[y])):
                    sig1 = signaturesWithClusters[z][j]
                    sig2 = signaturesWithClusters[y][n]
                    if sig1[1] != sig2[1]:
                        sig1mh = sig1[2]
                        sig2mh = sig2[2]

                        same = 0
                        for i in range(0, len(sig1mh)):
                            if sig1mh[i] == sig2mh[i]:
                                same += 1
                        jaccard = 1.0 - ((1.0 * same) / numHashes)

                        totalScores += jaccard

                        file.write("%s: cluster=%s:%s vs cluster=%s:%s " % (jaccard, sig1[0], sig1[1], sig2[0], sig2[1]))
                        file.write('\n')

                        x += 1
            averageScore = totalScores / x
            averages.append(averageScore)
            file.write("AverageScore%sv%s: %s" % (z, y,averageScore))
            file.write('\n')
    i = 0
    for z in range(k):
        for y in range(z, k):
            file.write("AverageScore%sv%s: %s" % (z, y, averages[i]))
            file.write('\n')
            i += 1
    file.close()

'''-------------------------------------------------------------------driver---------------------------------------------------------'''
ReadDataSet()
CleanData()

#SupervisedLearning()
UnsupervisedLearning()

