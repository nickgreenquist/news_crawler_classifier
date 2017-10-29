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

#read tokens into category token lists
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

                #category.token_list.append(article[1])
                category.token_list.append(line)
        file.close()
print ("Total Articles: " + str(total_articles))
print ('\n')

#remove duplicate articles
total_articles = 0
for category in categories:

    removed = 0
    dupes = list(set([x for x in category.token_list if category.token_list.count(x) > 1]))
    for d in dupes:
        #print(d.split(":::::")[0])
        removed += 1
    print("Dupes in %s: %s" % (category.name, str(len(dupes))))

    category.token_list = list(set(category.token_list))
    total_articles += len(category.token_list)
print ("Total Articles with removed duplicates: " + str(total_articles))
print ('\n')

#trim out dataset
cachedStopWords = stopwords.words("english")
for category in categories:
    newList = []
    i = 0
    while i < len(category.token_list):
        category.token_list[i] = category.token_list[i].lower()

        #remove easy punctation
        category.token_list[i] = re.sub(r"[,.;@#?!&$-]+\ *", " ", category.token_list[i])   

        #remove punctuation
        category.token_list[i] = "".join(l for l in category.token_list[i] if l not in string.punctuation)   

        category.token_list[i] = re.sub('\s+',' ',category.token_list[i])

        #remove stop words
        text = ' '.join([word for word in category.token_list[i].split() if word not in cachedStopWords])
        category.token_list[i] = text

        '''#remove sentences without at least 5 words
        if len(category.token_list[i].split()) < 5:
            #print ("removing: " + category.token_list[i])
            del category.token_list[i]'''
        i += 1

#shuffle arrays
for category in categories:
    random.shuffle(category.token_list)

#find train and test sets for each category
for category in categories:
    c_length = len(category.token_list)
    category.train = category.token_list[: int(.8 *c_length)]
    category.test = category.token_list[int(.8 *c_length):c_length]

#Find train tokens from 80% of the token list per category
x_train_tokens = []
y_train_tags = []
for category in categories:
    for tok in category.train:
        x_train_tokens.append(tok)
        y_train_tags.append([category.name])
X_train = np.array(x_train_tokens)

'''---------------------------------------------OneVsRestClassifier--------------------------------------'''
'''print ("Results for OneVsRestClassifier")
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train_tags)
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(X_train, Y)

#Find test tokens from 20% of the token list per category
target_names = []
for category in categories:
    try:
        #grab test tokens for this category
        target_names.append(category.name)
        c_length = len(category.token_list)
        test_tokens = []
        Y_test = []
        for tok in category.test:
            test_tokens.append(tok)
            Y_test.append([category.name])

        #classify test tokens   
        X_test = np.array(test_tokens)
        predicted = classifier.predict(X_test)
        all_labels = mlb.inverse_transform(predicted)
        
        #display results
        total = 0
        correct = 0
        for item, labels in zip(X_test, all_labels):
            #since multiple labels are possible (as well as none), 
            #we check if the correct label is in the predicted label list
            if category.name in labels: 
                correct += 1
            total += 1
        print (category.name + ": " + str(correct / total))

    except:
        print(category.name + ": ERROR")
print ('\n')'''

'''---------------------------------------------OneVsOneClassifier--------------------------------------'''
print ("Results for OneVsOneClassifier")
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsOneClassifier(LinearSVC()))])
Y = np.array(y_train_tags)
classifier.fit(X_train, Y.ravel())

#Find test tokens from 20% of the token list per category
target_names = []
for category in categories:
    try:
        #grab test tokens for this category
        target_names.append(category.name)
        c_length = len(category.token_list)
        test_tokens = []
        Y_test = []
        for tok in category.test:
            test_tokens.append(tok)
            Y_test.append([category.name])

        #classify test tokens   
        X_test = np.array(test_tokens)
        predicted = classifier.predict(X_test)

        #results
        print (category.name + ": " + str(np.mean(predicted == np.array(Y_test))))

    except:
        print(category.name + ": ERROR")
print('\n')

'''-------------------------------------------Naives Bayes----------------------------------------------------'''
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
clf = MultinomialNB().fit(X_train_tfidf, Y.ravel())

for category in categories:
    c_length = len(category.token_list)
    test_tokens = []
    Y_test = []
    for tok in category.test:
            test_tokens.append(tok)
            Y_test.append([category.name])


    X_new_counts = count_vect.transform(test_tokens)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    #results
    print (category.name + ": " + str(np.mean(predicted == np.array(Y_test))))
print ('\n')

'''-------------------------------------------Support Vector Machine----------------------------------------------------'''
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
    c_length = len(category.token_list)
    test_tokens = []
    Y_test = []
    for tok in category.test:
            test_tokens.append(tok)
            Y_test.append([category.name])

    predicted = text_clf.predict(test_tokens)
    
    #results
    print (category.name + ": " + str(np.mean(predicted == np.array(Y_test))))