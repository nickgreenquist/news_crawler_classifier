import csv
import random
import math
from math import sqrt
import string
import numpy
import binascii
import re
import csv 

STOP_WORDS = set([u'strong', u'all', u'just', u'being', u'over', u'both', u'through', u'yourselves', u'its',\
                 u'before', u'o', u'hadn', u'herself', u'll', u'had', u'should', u'to', u'only', \
                 u'won', u'under', u'ours', u'has', u'do', u'them', u'his', u'very', u'they', u'not',\
                 u'during', u'now', u'him', u'nor', u'd', u'did', u'didn', u'this', u'she', u'each',\
                 u'further', u'where', u'few', u'because', u'doing', u'some', u'hasn', u'are', u'our',\
                 u'ourselves', u'out', u'what', u'for', u'while', u're', u'does', u'above', u'between',\
                 u'mustn', u't', u'be', u'we', u'who', u'were', u'here', u'shouldn', u'hers', u'by', u'on',\
                 u'about', u'couldn', u'of', u'against', u's', u'isn', u'or', u'own', u'into', u'yourself',\
                 u'down', u'mightn', u'wasn', u'your', u'from', u'her', u'their', u'aren', u'there', u'been',\
                 u'whom', u'too', u'wouldn', u'themselves', u'weren', u'was', u'until', u'more', u'himself',\
                 u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me', u'myself', u'ma', u'these',\
                 u'up', u'will', u'below', u'ain', u'can', u'theirs', u'my', u'and', u've', u'then', u'is', u'am',\
                 u'it', u'doesn', u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if', u'again', u'no',\
                 u'when', u'same', u'how', u'other', u'which', u'you', u'shan', u'needn', u'haven', u'after',\
                 u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'yours', u'so', u'y', u'the', u'having', u'once'])

numHashes = 100
nextPrime = 4294967311
maxShingleID = 2**32-1
iterations = 30
#k = 4

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

def mapper():
    kvp = []
    with open("inputs/Articles.csv") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    print(len(content))
    for c in content:
        #csv = c.split('\n')
        #csv = re.split(r',business|,sports',c)
        #for line in csv:
        #pair = line.split("\",")

        pair = re.split(r'\",[0-9]+',c)
        if len(pair) > 1:
            article = pair[0]
            info = pair[1].split(",")
            if len(info) > 1:
                heading = info[1]
                #if len(heading.split()) > 2 and len(heading.split()) < 11:
                if True:
                    article = article.lower()
                    article = "".join(l for l in article if l not in string.punctuation)
                    article = re.sub(r'[^a-zA-Z ]', '', article)
                    article = article.split()
                    article = list(set(article))
                    article = [token for token in article if not token in STOP_WORDS]
                    slist = []
                    for i in range(0, len(article)):
                        shingle = article[i]
                        crc = shingle
                        crc = hash(shingle)
                        #crc = hash(shingle) % nextPrime
                        #crc = binascii.crc32(shingle) & 0xffffffff
                        slist.append(crc)
                    kvp.append((heading, slist))

    
    numCols = len(kvp)
    return kvp

def hasher(kvp):
    signatures = []
    # For each document...
    print(len(kvp))
    for doc in kvp:
    
        # Get the shingle set for this document.
        shingleIDSet = doc[1]
        
        # The resulting minhash signature for this document. 
        signature = []
        
        # For each of the random hash functions...
        for i in range(0, numHashes):
            
            # For each of the shingles actually in the document, calculate its hash code
            # using hash function 'i'. 
            
            # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
            # the maximum possible value output by the hash.
            minHashCode = nextPrime + 1
            
            # For each shingle in the document...
            for shingleID in shingleIDSet:
                # Evaluate the hash function.
                hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime 
                hashCode = hashCode / nextPrime
                
                # Track the lowest hash code seen.
                if hashCode < minHashCode:
                    minHashCode = hashCode

            # Add the smallest hash code value as component number 'i' of the signature.
            signature.append(minHashCode)
        
        # Store the MinHash signature for this document.
        signatures.append((doc[0], signature))
    
    return signatures

def dist(x, c):
    """Euclidean distance between sample x and cluster center c.
    Inputs: x, a sparse vector
            c, a dense vector
    """
    sqdist = 0.
    i = 0
    for v in x:
        sqdist += (v - c[i]) ** 2
        i += 1
    return sqrt(sqdist)

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

def calcmean(clusterMembers):
    """Mean (as a dense vector) of a set of sparse vectors of length l."""
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
    centersFromSig = random.sample(signatures, k)
    centers = []
    for i in range(k):
        centers.append(centersFromSig[i][1])
    '''for sig in signatures:
        if "World Cup" in sig[0]:
            centers[0] = sig[1]
        if "stocks" in sig[0]:
            centers[1] = sig[1]'''
    print(centers)

    cluster = [None] * len(signatures)

    print("%s: %s" % ("Iteration: ", iterations))
    for _ in range(0, iterations):
        print("%s: %s" % ("iter: ", _))
        i = 0
        for sig in signatures:
            cluster[i] = (sig[0], min(range(k), key=lambda j: dist(sig[1], centers[j])))
            #cluster[i] = (sig[0], min(range(k), key=lambda j: jacdist(sig[1], centers[j])))
            i += 1
        j = 0
        for c in enumerate(centers):
            clusterMembers = (x for i, x in enumerate(signatures) if cluster[i][1] == j)
            #clusterMemers = list(clusterMembers)
            temp = calcmean(clusterMembers)
            centers[j] = temp
            j += 1
    return cluster, centers



'''read in csv data by col'''
kvp = mapper()

'''get signature matrix'''
signatures = hasher(kvp)

'''print out signature matrix'''
file = open("results/signatures.txt", "w")
for sig in signatures:
    try:
        file.write("%s,%s" % (sig[0], sig[1]))
        file.write('\n')
    except:
        file.write("ERROR")
file.close()

'''PERFORM KMEANS CLUSTERING'''
#for k in range(2, 3):
k = 2
kmeansReturn  = kmeans(signatures, k)
centers = kmeansReturn[1]
cluster_ind = kmeansReturn[0]

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


'''COMPUTE JACCARD DISTANCES'''
count = 50
for i in range(k):
    signaturesWithClusters[i] = random.sample(signaturesWithClusters[i], count)

'''file = open("results/jaccard.txt", "w")
for z in range(k):
    for y in range(z, k):
        print("z:%s, y:%s" % (z,y))
        averageScore = 0
        totalScores = 0
        x = 0
        j = 0
        for _ in range(0, 49):
            sig1 = signaturesWithClusters[z][j]
            sig2 = signaturesWithClusters[y][j + 1]
            sig1mh = sig1[2]
            sig2mh = sig2[2]
            words = []
            for w in sig1mh:
                words.append(w)
            for w in sig2mh:
                words.append(w)
            words = set(words)
            words = list(words)
            total = len(words)

            same = 0
            for i in range(0, len(sig1mh)):
                if sig1mh[i] == sig2mh[i]:
                    same += 1
            jaccard = 1 - (same / total)

            totalScores += jaccard

            file.write("%s" % (jaccard))
            #file.write("%s: cluster=%s:%s vs cluster=%s:%s " % (jaccard, sig1[0], sig1[1], sig2[0], sig2[1]))
            file.write('\n')
            x += 1
            j += 2

        averageScore = totalScores / x
        file.write("AverageScore%sv%s: %s" % (z, y,averageScore))
        file.write('\n')
file.close()'''

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
                    words = []
                    for w in sig1mh:
                        words.append(w)
                    for w in sig2mh:
                        words.append(w)
                    words = set(words)
                    words = list(words)
                    total = len(words)

                    same = 0
                    for i in range(0, len(sig1mh)):
                        if sig1mh[i] == sig2mh[i]:
                            same += 1
                    jaccard = 1 - (same / total)

                    totalScores += jaccard

                    #file.write("%s" % (jaccard))
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
