# news_crawler_classifier
A Python program to crawl RSS feeds and use sklearn classification algorithms on the results

Instroduction:
This program contains 2 .py files. Two are crawlers and one is a classification program. The crawler looks for 6 categories of news articles. The crawler_multi.py will read on average 2000 articles. These articles are parsed by the newspaper library. The articles are saved in a sub folder corresponding to today's date.

The classifier.py program will read txt files line by line from any subfiles inside a directory passed in from the command line. If no directory is passed in, they default to data. The classifier will recurse through the directory and pick up all unique text file names. These file names will become the category names that will be used for classification. If the standard data directory is passed in, 6 categories will be generated: Business, Arts and Entertainment, Politics, Science and Health, Sports, and Technology. 

The classifier trains 4 classificaiton models using a 80/20 split of a randomly shuffled token list for each category. NOTE: A token is an article. The first two algorithms are OneVsRest and OneVsOne. The second and third algorithms are NaiveBayes and Support Vector Machine. Each algorithm is trained with the same random 80% of the token_list for each category. The remaining 20% will be used as the test set for each algorithm. NOTE: Shuffling only occurs once so each 80/20 split is the same for each run so algorithm performance can be compared. 

Before classification, preprocessing is done on every token in every article. Puncuation is removed, spaces are condensed to one, all words are set to lowercase, and english stop words are removed. 

How to Run:
1. Open the command prompt in the news_crawler_classifier folder
2. Run a crawl command
  2a. python crawler.py [optional output directory]
NOTE: the crawler.py will take about 20 minutes to run

3. Run the classifier command
  3a. python classifier.py [optional input directory]
 NOTE: the input directory will default to data folder
 
 For Better Results:
  You can crawl the links for a few days to build the data set. Everytime you crawl on a new day a new subfolder is created with the date so old data will not be overwritten. In fact, it is recommended that a few days worth of data is collected before classifying. Also, repeated articles are filtered out as some rss feeds don't just show articles from a specific day
