#Imports for crawling libraries
import feedparser
import urllib
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article
import string
import re
from nltk import tokenize
import sys
import os
import errno
import datetime as dt  

'''------------------------------------Crawling-----------------------------------'''
class Category():
    def __init__(self, name, links, token_list, articles):
        self.name = name
        self.links = links
        self.token_list = token_list #a token is a sentence
        self.articles = articles

categories = []

us = Category(name = 'US', links = [], token_list = [], articles = [])
world = Category(name = 'World', links = [], token_list = [], articles = [])

us.links.append('http://rss.cnn.com/rss/cnn_us.rss')
'''us.links.append('http://feeds.reuters.com/Reuters/domesticNews')
us.links.append('http://rss.nytimes.com/services/xml/rss/nyt/US.xml')
us.links.append('http://abcnews.go.com/abcnews/usheadlines')
us.links.append('http://feeds.foxnews.com/foxnews/national')
us.links.append('https://www.cnbc.com/id/100727362/device/rss/rss.html')

world.links.append('http://rss.cnn.com/rss/cnn_world.rss')
world.links.append('http://feeds.reuters.com/Reuters/worldNews')
world.links.append('http://rss.nytimes.com/services/xml/rss/nyt/World.xml')
world.links.append('http://abcnews.go.com/abcnews/internationalheadlines')
world.links.append('http://feeds.foxnews.com/foxnews/world')
world.links.append('https://www.cnbc.com/id/100727362/device/rss/rss.html')'''

categories.append(us)
categories.append(world)

for category in categories:
    for link in category.links:
        print ("processing this link: " + link + " for " + category.name)
        try:
            feed = feedparser.parse(link)
            for entry in feed['entries']:
                try:
                    article = Article(entry['link'])
                    article.download()
                    article.parse()

                    #filter out bad characters
                    text = ''.join([c for c in article.text if c in string.printable])

                    #split up into tokens (sentences)
                    tokens = tokenize.sent_tokenize(text)

                    for tok in tokens:
                        tok = tok.strip()
                        if len(tok) > 5:
                            category.token_list.append(tok)
                    
                    #add entire article
                    text = re.sub('\s+',' ',text)
                    category.articles.append(text)
                except:
                    print ("ERROR: Failed with article link: " + entry['link'])
        except:
                print ("ERROR: Failed with rss link: " + link)


#Write all Tokens and Articles to File
total_sentences = 0
data_dir = os.getcwd() + '/data_usworld/' + dt.datetime.today().strftime("%Y-%m-%d")
if len(sys.argv) > 1:
    data_dir = os.getcwd() + '/' + sys.argv[1] + '/' + dt.datetime.today().strftime("%Y-%m-%d")
try:
    os.makedirs(data_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
for category in categories:
    total_sentences += len(category.token_list)
    print ("Category: " + category.name + " | Number of sentences: " + str(len(category.token_list)))

    #write sentences to file
    file = open((data_dir + "/" + "/" + category.name + ".txt"),"w") 
    for token in category.token_list:
        file.write(token + '\n')
    file.close()
print ("Total Sentences: " + str(total_sentences))