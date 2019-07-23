# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 20:03:03 2019

@author: prach
"""
import csv
import math
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
en_stops = set(stopwords.words('english'))
ps = PorterStemmer()
c_words=[]
cc_words=[]
file = open("test4.csv")
reader = csv.reader(file)

data = [
    [(word.replace(",", "")
          .replace(".", "")
          .replace("(", "")
          .replace(")", ""))
    for word in row[1].lower().split()]
    for row in reader]
file.close()
file = open("test4.csv")
reader_patent = csv.reader(file)
patent_num=[]

for row in reader_patent:
    patent_num.append(row[0])
patent_num=patent_num[1:]
print(patent_num)

data = data[1:]
for words in data: 
    c_words=[]
    for word in words:
        if word not in en_stops:
            word = ps.stem(word)
            c_words.append(word)
    cc_words.append(c_words)

  #Removes header

#print("========================DATA========================================")
#print(cc_words)
#print("========================DATA END========================================")

def computeReviewTFDict(abstract):
    """ Returns a tf dictionary for each review whose keys are all 
    the unique words in the review and whose values are their 
    corresponding tf.
    """
    #Counts the number of times the word appears in review
    abstractTFDict = {}
    for word in abstract:
        if word in abstractTFDict:
            abstractTFDict[word] += 1
        else:
            abstractTFDict[word] = 1
    #Computes tf for each word           
    for word in abstractTFDict:
        abstractTFDict[word] = abstractTFDict[word] / len(abstract)
    return abstractTFDict
TFDict=[]
for d in cc_words:
    TFDict.append(computeReviewTFDict(d))
#print("========================TF DICT========================================")
#print(TFDict)
#print("========================TF DICT END========================================")

def computeCountDict():
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    countDict = {}
    # Run through each review's tf dictionary and increment countDict's (word, doc) pair
    for review in TFDict:
        for word in review:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict

  #Stores the review count dictionary
countDict = computeCountDict()

def computeIDFDict():
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in countDict:
        idfDict[word] = math.log(len(data) / countDict[word])
    return idfDict
  
  #Stores the idf dictionary
idfDict = computeIDFDict()

#print("========================IDF DICT========================================")
#print(idfDict)
#print("========================IDF DICT END========================================")

def computeReviewTFIDFDict(reviewTFDict):
    """ Returns a dictionary whose keys are all the unique words in the
    review and whose values are their corresponding tfidf.
    """
    reviewTFIDFDict = {}
    #For each word in the review, we multiply its tf and its idf.
    for word in reviewTFDict:
        reviewTFIDFDict[word] = reviewTFDict[word] * idfDict[word]
    return reviewTFIDFDict

  #Stores the TF-IDF dictionaries
tfidfDict = [computeReviewTFIDFDict(review) for review in TFDict]

#print("========================TF-IDF DICT========================================")
#print(tfidfDict)
#print("========================TF-IDF DICT END=======================================")
 # Create a list of unique words
wordDict = sorted(countDict.keys())
def computeTFIDFVector(review):
    tfidfVector = [0.0] * len(wordDict)
     
      # For each unique word, if it is in the review, store its TF-IDF value.
    for i, word in enumerate(wordDict):
        if word in review:
            tfidfVector[i] = review[word]
    return tfidfVector

tfidfVector = [computeTFIDFVector(review) for review in tfidfDict]
#print("========================TF-IDF Vector========================================")
#print(tfidfVector)
#print("========================TF-IDF Vector END=======================================")

def dot_product(vector_x, vector_y):
    dot = 0.0
    for e_x, e_y in zip(vector_x, vector_y):
       dot += e_x * e_y
    return dot

def magnitude(vector):
    mag = 0.0
    for index in vector:
      mag += math.pow(index, 2)
    return math.sqrt(mag)

def similarityMatrixCalculator():
    review_similarity = np.zeros(shape=(10,10))
    for i in range(10):
        for j in range(10):
            review_similarity[i][j] = float(dot_product(tfidfVector[i], tfidfVector[j])/ magnitude(tfidfVector[i]) * magnitude(tfidfVector[j]))
    #print("Similairity is",review_similarity)
    return review_similarity
    
end_matrix=similarityMatrixCalculator()
print(end_matrix)
df = pd.DataFrame(end_matrix, columns=patent_num[:10], index=patent_num[:10])
df.to_csv('similarity_matrix.csv') 