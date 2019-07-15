# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 20:03:03 2019

@author: prach
"""
import csv
import math
file = open("Dataset.csv")
reader = csv.reader(file)

data = [
    [(word.replace(",", "")
          .replace(".", "")
          .replace("(", "")
          .replace(")", ""))
    for word in row[1].lower().split()]
    for row in reader]
    
  #Removes header
data = data[1:]
print("========================DATA========================================")
print(data)
print("========================DATA END========================================")

def computeReviewTFDict(review):
    """ Returns a tf dictionary for each review whose keys are all 
    the unique words in the review and whose values are their 
    corresponding tf.
    """
    #Counts the number of times the word appears in review
    reviewTFDict = {}
    for word in review:
        if word in reviewTFDict:
            reviewTFDict[word] += 1
        else:
            reviewTFDict[word] = 1
    #Computes tf for each word           
    for word in reviewTFDict:
        reviewTFDict[word] = reviewTFDict[word] / len(review)
    return reviewTFDict
TFDict=[]
for d in data:
    TFDict.append(computeReviewTFDict(d))
print("========================TF DICT========================================")
print(TFDict)
print("========================TF DICT END========================================")

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

print("========================IDF DICT========================================")
print(idfDict)
print("========================IDF DICT END========================================")

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

print("========================TF-IDF DICT========================================")
print(tfidfDict)
print("========================TF-IDF DICT END=======================================")
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
print("========================TF-IDF Vector========================================")
print(tfidfVector)
print("========================TF-IDF Vector END=======================================")

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


review_similarity = dot_product(tfidfVector[2], tfidfVector[1])/ magnitude(tfidfVector[2]) * magnitude(tfidfVector[1])
print("Similairity is",review_similarity)
    