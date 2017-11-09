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

filters = [
    'Please verify you\'re not a robot by clicking the box',
    'Please verify you\'re not a robot by clicking the box.',
    'Invalid email address.',
    'Please re-enter.',
    'You must select a newsletter to subscribe to.',
    'Sign Up You agree to receive occasional updates and special offers for The New York Times\'s products and services.',
    'Thank you for subscribing.',
    'An error has occurred.',
    'Please try again later.',
    'View all New York Times newsletters.',
    'Photo',
    'Advertisement Continue reading the main story',
    'Advertisement',
]

class ArticleCrawled():
    def __init__(self, heading, text, link):
        self.heading = heading
        self.text = text
        self.link = link

class Category():
    def __init__(self, name, links, token_list, articles):
        self.name = name
        self.links = links
        self.token_list = token_list
        self.articles = articles

categories = []
rss_links = []

def loadLinks():
    data_dir = os.getcwd() + '/links'
    if len(sys.argv) > 1:
        data_dir = os.getcwd() + '/' + sys.argv[1]
    for filename in os.listdir(data_dir):
        catname = filename.split('.')[0]
        if catname != 'rss':
            new_category = Category(name = filename.split('.')[0], links = [], token_list = [], articles = [])

        file = open((data_dir + '/' + filename),"r") 
        for line in file: 
            if catname == 'rss':
                rss_links.append(line)
            else:
                new_category.links.append(line)
        file.close()

        if catname != 'rss':
            categories.append(new_category)


def parseRSSPages():
    #loop through master RSS feed lists that grab all links from children rss links
    for rss_link in rss_links:
        html_page = urllib.request.urlopen(rss_link)
        soup = BeautifulSoup(html_page)
        links = soup.findAll('a', attrs={'href': re.compile("^http://")})

        #delete duplicate CNN and ABC links
        if "cnn" in rss_link or "abc" in rss_link:
            del links[::2]

        #extract actual href links
        for link in links:
            l = link.get('href')

            #filter out stuff we don't want (Yahoo might lead to duplicate articles)
            #money.cnn.com is another top level rss feed page that we don't want to add as an rss link
            if "yahoo" not in l and "video" not in l and "money.cnn.com" not in l:
                for c in categories:
                    name = c.name.lower()
                    if ("showbiz" in l or "entertainment" in l or "arts" in l) and "entertainment" in name:
                        c.links.append(l)
                    if ("health" in l) and "health" in name:
                        c.links.append(l)
                    if ("business" in l or "money" in l) and "business" in name:
                        c.links.append(l)
                    if ("politics" in l) and "politics" in name:
                        c.links.append(l)
                    if ("science" in l) and "science" in name:
                        c.links.append(l)
                    if ("sports" in l) and "sports" in name:
                        c.links.append(l)
                    if ("tech" in l) and "tech" in name:
                        c.links.append(l)

def parseArticles():
    for category in categories:
        for link in category.links:
            try:
                feed = feedparser.parse(link)
                print("%s - Processing: %s" % (category.name, link))
                for entry in feed['entries']:
                    try:
                        article = Article(entry['link'])
                        article.download()
                        article.parse()

                        heading = article.title
                        #print ("processing this article: " + heading)

                        #filter out bad characters
                        rawtext = ''.join([c for c in article.text if c in string.printable])

                        #split up into tokens (sentences)
                        tokens = tokenize.sent_tokenize(rawtext)

                        #filter out bad tokens
                        tokens = filterArticle(tokens)

                        articleText = ""
                        for tok in tokens:
                            articleText += tok
                            category.token_list.append(tok)
                        
                        #remove any extra whitespace from entire article
                        articleText = re.sub('\s+',' ',articleText)

                        newArticle = ArticleCrawled(heading = heading, text = articleText, link = entry['link'])

                        category.articles.append(newArticle)
                    except Exception as e:
                        print ("ERROR: Failed with article link: " + entry['link'])
                        print(e)
            except Exception as e:
                    print ("ERROR: Failed with rss link: " + link)
                    print(e)

def filterArticle(tokens):
    newTokens = []

    for token in tokens:
        token = token.strip()
        token = re.sub('\s+',' ',token)

        valid = True
        for f in filters:
            if f.lower() == token.lower():
                valid = False

        if valid:
            newTokens.append(token)

    return newTokens


def writeData():
    #Write complete articles to files
    data_dir = os.getcwd() + '/data/' + dt.datetime.today().strftime("%Y-%m-%d")
    if len(sys.argv) > 1:
        data_dir = os.getcwd() + '/' + sys.argv[1] + '/' + dt.datetime.today().strftime("%Y-%m-%d")
    try:
        os.makedirs(data_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for category in categories:
        #write articles to file
        file = open((data_dir + "/" + "/" + category.name + "_articles.txt"),"w") 
        for article in category.articles:
            try:
                #file.write(article.link + ':::::' + article.heading + ':::::' + article.text)
                file.write(article.heading + ':::::' + article.text)
                file.write('\n')
            except Exception as e:
                    print(e)
        file.close()

        #write sentences to file
        file = open((data_dir + "/" + "/" + category.name + ".txt"),"w") 
        for token in category.token_list:
            try:
                file.write(token + '\n')
            except Exception as e:
                    print(e)
        file.close()

loadLinks()
parseRSSPages()
parseArticles()
writeData()