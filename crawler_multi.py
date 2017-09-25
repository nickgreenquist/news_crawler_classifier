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

business = Category(name = 'Business', links = [], token_list = [], articles = [])
entertainment_art = Category(name = 'Entertainment and Arts', links = [], token_list = [], articles = [])
health = Category(name = 'Health', links = [], token_list = [], articles = [])
politics = Category(name = 'Politics', links = [], token_list = [], articles = [])
science = Category(name = 'Science', links = [], token_list = [], articles = [])
sports = Category(name = 'Sports', links = [], token_list = [], articles = [])
tech = Category(name = 'Tech', links = [], token_list = [], articles = [])

#Sites with many links to seperate RSS feeds
rss_links = []
'''rss_links.append('https://www.reuters.com/tools/rss')
rss_links.append('http://www.cnn.com/services/rss/')
rss_links.append('http://www.nbcnewyork.com/rss/')
rss_links.append('http://abcnews.go.com/Site/page/rss--3520115')
rss_links.append('https://www.wired.com/about/rss_feeds/')
rss_links.append('http://www.bbc.com/news/10628494')'''

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
        if "yahoo" not in l and "video" not in l:
            if ("showbiz" in l or "entertainment" in l or "arts" in l): entertainment_art.links.append(l)
            if ("business" in l or "money" in l): business.links.append(l)
            if ("health" in l): health.links.append(l)
            if ("politics" in l): politics.links.append(l)
            if ("science" in l): science.links.append(l)
            if ("sports" in l): sports.links.append(l)
            if ("tech" in l): tech.links.append(l)

#Add some standard rss links for specific categories
#We are doing this becuase some master lists don't play nice when opened from above loop
business.links.append('http://www.economist.com/sections/business-finance/rss.xml')
'''business.links.append('http://nypost.com/business/feed/')
business.links.append('https://www.cnbc.com/id/10001147/device/rss/rss.html')
business.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Business.xml')
business.links.append('http://rss.nytimes.com/services/xml/rss/nyt/SmallBusiness.xml')

entertainment_art.links.append('http://feeds.foxnews.com/foxnews/entertainment')
entertainment_art.links.append('http://www.smithsonianmag.com/rss/arts-culture/')
entertainment_art.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Arts.xml')
entertainment_art.links.append('http://rss.nytimes.com/services/xml/rss/nyt/ArtandDesign.xml')
entertainment_art.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Movies.xml')
entertainment_art.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Theater.xml')

health.links.append('http://feeds.foxnews.com/foxnews/health')
health.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Health.xml')
health.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Nutrition.xml')
health.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Research.xml')

politics.links.append('http://feeds.foxnews.com/foxnews/politics')
politics.links.append('https://www.cnbc.com/id/10000113/device/rss/rss.html')
politics.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Politics.xml')
politics.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Upshot.xml')

science.links.append('http://feeds.foxnews.com/foxnews/science')
science.links.append('https://www.sciencedaily.com/rss/top/science.xml')
science.links.append('https://www.eurekalert.org/rss.xml')
science.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Science.xml')

sports.links.append('http://feeds.foxnews.com/foxnews/sports')
sports.links.append('http://www.espn.com/espn/rss/news')
sports.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Sports.xml')
sports.links.append('http://rss.nytimes.com/services/xml/rss/nyt/ProFootball.xml')
sports.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Baseball.xml')

tech.links.append('http://feeds.feedburner.com/TechCrunch/')
tech.links.append('http://feeds.foxnews.com/foxnews/tech')
tech.links.append('https://www.cnbc.com/id/19854910/device/rss/rss.html')
tech.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Technology.xml')
tech.links.append('http://rss.nytimes.com/services/xml/rss/nyt/PersonalTech.xml')'''

categories.append(business)
categories.append(entertainment_art)
categories.append(health)
categories.append(politics)
categories.append(science)
categories.append(sports)
categories.append(tech)

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
data_dir = os.getcwd() + '/data_multi/' + dt.datetime.today().strftime("%Y-%m-%d")
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