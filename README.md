# rss_classifier
A Python program to crawl RSS feeds and use sklearn classification algorithms on the results

Instroduction:
This program contains 3 .py files. Two are crawlers and one is a classification program. One crawler looks for 7 categories of news articles. The other looks for only 2 (world and US). Both are preloaded with many rss links that the program will crawl over. The crawler_multi.py will read on average 55,000 sentences from news articles. These sentences are parsed by the newspaper library and the crawler also tosses sentences less than 5 characters in length. The crawler_usworld.py will read on average 8000 sentences per run. Both crawlers save their sentence tokens in txt files located in data_multi and data_usworld respectively. The results are saved in a sub folder corresponding to today's date.

The classifier.py program will read txt files line by line from any subfiles inside a directory passed in from the command line. If no directory is passed in, they default to data_multi and data_usworld respectively. The classifier will recurse through the directory and pick up all unique text file names. These file names will become the category names that will be used for classification. NOTE: It is recommended not to mix the US and World text files with the text files generated from crawler_multi.py. If the classifier is passed in the data_usworld directory, two categories will be generated (US and World). If the data_multi directory is passed in, 7 categories will be generated: Business, Arts and Entertainment, Health, Politics, Science, Sports, and Technology. 

The classifier trains 4 classificaiton models using a 80/20 split of a randomly shuffled token list for each category. NOTE: A token is a sentence. The first two algorithms are OneVsRest and OneVsOne. The second and third algorithms are NaiveBayes and Support Vector Machine. Each algorithm is trained with the same random 80% of the token_list for each category. The remaining 20% will be used as the test set for each algorithm. NOTE: Shuffling only occurs once so each 80/20 split is the same for each run so algorithm performance can be compared safely. 

Before classification, preprocessing is done on every token in every category. Puncuation is removed, spaces are condensed to one, all words are set to lowercase, sentences with less than 5 words in them are tossed out, and english stop words are removed. 

How to Run:
1. Open the command prompt in the RSSClassifier folder
2. Run a crawl command
  2a. python crawler_multi.py [optional output directory]
  2b. python crawler_usworld.py [optional ouput directory]
NOTE: the crawler_multi.py will take about 15 minutes to run
NOTE: the output directory will default to data_multi and data_usworld respectively

3. Run the classifier command
  3a. python classifier.py [optional input directory]
 NOTE: the input directory will default to data_multi and data_usworld respectively
 
 For Better Results:
  You can crawl the links for a few days to build the data set. Everytime you crawl on a new day a new subfolder is created with the date so old data will not be overwritten. In fact, it is recommended that a few days worth of data is collected before classifying.
