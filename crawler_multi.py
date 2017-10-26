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
    def __init__(self, heading, text):
        self.heading = heading
        self.text = text

class Category():
    def __init__(self, name, links, token_list, articles):
        self.name = name
        self.links = links
        self.token_list = token_list
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
rss_links.append('https://www.reuters.com/tools/rss')
rss_links.append('http://www.cnn.com/services/rss/')
rss_links.append('http://www.nbcnewyork.com/rss/')
rss_links.append('http://abcnews.go.com/Site/page/rss--3520115')
rss_links.append('http://www.bbc.com/news/10628494')


#Add some standard rss links for specific categories
#We are doing this becuase some master lists don't play nice when opened from above loop
'''business.links.append('http://www.economist.com/sections/business-finance/rss.xml')
business.links.append('http://nypost.com/business/feed/')
business.links.append('https://www.cnbc.com/id/10001147/device/rss/rss.html')
business.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Business.xml')
business.links.append('https://rss.upi.com/news/business_news.rss')
business.links.append('http://rss.cnn.com/rss/money_news_economy.rss')
business.links.append('http://rss.cnn.com/rss/money_markets.rss')

entertainment_art.links.append('http://feeds.foxnews.com/foxnews/entertainment')
entertainment_art.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Arts.xml')
entertainment_art.links.append('http://rss.nytimes.com/services/xml/rss/nyt/ArtandDesign.xml')
entertainment_art.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Movies.xml')
entertainment_art.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Theater.xml')
entertainment_art.links.append('https://rss.upi.com/news/entertainment_news.rss')'''

health.links.append('http://feeds.foxnews.com/foxnews/health')
health.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Health.xml')
health.links.append('https://rss.upi.com/news/health_news.rss')

'''politics.links.append('http://feeds.foxnews.com/foxnews/politics')
politics.links.append('https://www.cnbc.com/id/10000113/device/rss/rss.html')
politics.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Politics.xml')
politics.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Upshot.xml')
politics.links.append('http://feeds.washingtonpost.com/rss/politics')'''

science.links.append('http://feeds.foxnews.com/foxnews/science')
science.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Science.xml')
science.links.append('https://rss.upi.com/news/science_news.rss')
science.links.append('http://feeds.latimes.com/latimes/news/science')
science.links.append('http://rss.sciam.com/ScientificAmerican-News')

'''sports.links.append('http://feeds.foxnews.com/foxnews/sports')
sports.links.append('http://www.espn.com/espn/rss/news')
sports.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Sports.xml')
sports.links.append('http://rss.nytimes.com/services/xml/rss/nyt/ProFootball.xml')
sports.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Baseball.xml')
sports.links.append('https://www.wired.com/feed/category/science/latest/rss')

tech.links.append('http://feeds.feedburner.com/TechCrunch/')
tech.links.append('http://feeds.foxnews.com/foxnews/tech')
tech.links.append('https://www.cnbc.com/id/19854910/device/rss/rss.html')
tech.links.append('http://rss.nytimes.com/services/xml/rss/nyt/Technology.xml')
tech.links.append('http://rss.nytimes.com/services/xml/rss/nyt/PersonalTech.xml')
tech.links.append('http://www.techradar.com/rss')
tech.links.append('https://www.cnet.com/g00/3_c-6bbb.hsjy.htr_/c-6RTWJUMJZX77x24myyux78x3ax2fx2fbbb.hsjy.htrx2fwx78x78x2fsjbx78x2f_$/$/$/$')
tech.links.append('https://www.techrepublic.com/rssfeeds/articles/')
tech.links.append('http://rssfeeds.usatoday.com/usatoday-TechTopStories')'''

categories.append(business)
categories.append(entertainment_art)
categories.append(health)
categories.append(politics)
categories.append(science)
categories.append(sports)
categories.append(tech)

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
            if "yahoo" not in l and "video" not in l and "money.cnn.com" not in l and ("health" in l or "science" in l):
                if ("showbiz" in l or "entertainment" in l or "arts" in l): entertainment_art.links.append(l)
                if ("health" in l): health.links.append(l)
                if ("business" in l or "money" in l): business.links.append(l)
                if ("politics" in l): politics.links.append(l)
                if ("science" in l): science.links.append(l)
                if ("sports" in l): sports.links.append(l)
                if ("tech" in l): tech.links.append(l)

def parseArticles():
    for category in categories:
        for link in category.links:
            try:
                feed = feedparser.parse(link)
                print("Processing: " + link)
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

                        newArticle = ArticleCrawled(heading = heading, text = articleText)

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

            file.write(article.heading + ':::::' + article.text)
            file.write('\n')
        file.close()

        #write sentences to file
        file = open((data_dir + "/" + "/" + category.name + ".txt"),"w") 
        for token in category.token_list:
            file.write(token + '\n')
        file.close()

parseRSSPages()
parseArticles()
writeData()