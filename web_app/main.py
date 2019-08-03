"""
Created on Tue Jul  23 17:02:27 2019

@author: prach
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
patentlist = pd.read_csv('../data/Dataset.csv')

patentlist.head()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
cachedStopWords = set(stopwords.words("english"))
ps = PorterStemmer()

from collections import Counter
def generateTFIDFMatrix():
    patentlist['abstract_org']=patentlist['abstract']
    abstract=[]
    for item in patentlist['abstract_org']:
        text=[]
        strtext=''
        for word in item.lower().split():
            word=word.replace(",", "").replace(".", "").replace("(", "").replace(")", "")
            if word not in cachedStopWords:
                word = ps.stem(word)
                text.append(word)
            strtext=' '.join(text)
        abstract.append(strtext)
    patentlist['abstract_org']=abstract
    freq_count = []
    for item in patentlist['abstract_org']:
        count = Counter(str(item).split())
        freq_count.append(count)
    patentlist['word_count'] = freq_count
    tfidfVect = TfidfVectorizer()
    tfidf = tfidfVect.fit_transform(patentlist['abstract_org'])
    pickle.dump(tfidf, open("tfidf_fit_transform.pickle", "wb"))
    
def generateFitVector():
    patent1_tfidfVect = TfidfVectorizer()
    patent1_tfidfVect = patent1_tfidfVect.fit(patentlist['abstract_org'])
    pickle.dump(patent1_tfidfVect, open("tfidf_fit.pickle", "wb"))
    
def patentKeywordMatch(keyword):
    search=[]
    text=[]
    strtext=[]
    for word in keyword.lower().split():
            word=word.replace(",", "").replace(".", "").replace("(", "").replace(")", "")
            if word not in cachedStopWords:
                word = ps.stem(word)
                text.append(word)
            strtext=' '.join(text)
            search.append(strtext)
    patent1 =search
    patent1_tfidfVect = pickle.load(open("tfidf_fit.pickle", "rb"))
    patent1_tfidf = patent1_tfidfVect.transform(patent1)
    new_tfidf=pickle.load(open("tfidf_fit_transform.pickle", "rb"))
    nbrs = NearestNeighbors(n_neighbors=5).fit(new_tfidf)
    distances, indices = nbrs.kneighbors(patent1_tfidf)
    names_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['patent_num'])
    abstract_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['abstract'])
    url_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['url'])
    title_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['title'])
    result = pd.DataFrame({'distance':distances.flatten(), 'patent_num':names_similar, 'abstract':abstract_similar, 'url':url_similar,
                      'title':title_similar})
    result=result.sort_values('distance')
    result.to_csv('result_keyword.csv')
    return result

def patentPatentIdMatch(keyword):
    patent1 = patentlist[patentlist['patent_num'] == keyword]
    patent2=patent1['abstract']
    print(patent2)
    abstract=[]
    for item in patent2:
        text=[]
        strtext=''
        for word in item.lower().split():
            word=word.replace(",", "").replace(".", "").replace("(", "").replace(")", "")
            if word not in cachedStopWords:
                word = ps.stem(word)
                text.append(word)
            strtext=' '.join(text)
        abstract.append(strtext)
    tfidf = pickle.load(open("tfidf_fit_transform.pickle", "rb"))
    patent1 = patentlist[patentlist['patent_num'] == keyword]
    patent1=abstract
    patent1_tfidfVect = pickle.load(open("tfidf_fit.pickle", "rb"))
    patent1_tfidf = patent1_tfidfVect.transform(patent1)
    patent1_tfidfVect.vocabulary_
    patent1_tfidf_table = pd.DataFrame(sorted(patent1_tfidfVect.vocabulary_.items(),key=lambda pair: pair[1],reverse=True))
    patent1_tfidf_table
    feature_names = patent1_tfidfVect.get_feature_names()
    for col in patent1_tfidf.nonzero()[1]:
        print(feature_names[col], ' - ', patent1_tfidf[0, col])
    nbrs = NearestNeighbors(n_neighbors=6).fit(tfidf)
    distances, indices = nbrs.kneighbors(patent1_tfidf)
    names_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['patent_num'])
    abstract_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['abstract'])
    url_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['url'])
    title_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['title'])
    result = pd.DataFrame({'distance':distances.flatten(), 'patent_num':names_similar, 'abstract':abstract_similar, 'url':url_similar,
                      'title':title_similar})
    result=result.sort_values('distance')
    result.to_csv('result_patentid.csv')
    
    return result

def get_patentnumbers(id,df):
    subsetDataFrame = df['PATENT NUMBER'][df['UID'] == id].str.split(", ", expand = True) 
    return subsetDataFrame


def get_similarity_matrix(k):
    numbers = k[0]
    for i in range(0, len(numbers)): 
        numbers[i] = int(numbers[i])
    len(numbers)
    data= pd.read_csv('../data/similarity_matrix.csv')
    data.rename( columns={'Unnamed: 0':'patent_no'}, inplace=True)
    data.set_index('patent_no', inplace=True)

    results = list(data.columns.values) 
    results

    pp= data.loc[numbers]
    pp= pp.dropna()
    pp        

    demo=[]
    comb=[]
    for i in range(0,len(numbers)-1):
        for j in range(0,len(results)):
            comb=[]
            if pp.iloc[i][j]!=0.0:
                comb.append(pp.iloc[i][j])
                comb.append(results[j])
                demo.append(comb)

##Sort patents by similarity
    arr = sorted(demo,reverse=True)[:30]
    arr1=[]
    flag=False
    for i in range(0, len(arr)):
        flag=False
        for j in range(0,len(arr1)):
            if arr[i][1]==arr1[j][1]:
                flag=True
        if(flag==False):
            arr1.append(arr[i])
    arr1=arr1[:6]
    
    patentlist = pd.read_csv('../data/Dataset.csv')
    patent1 = patentlist[patentlist['patent_num'] == str(arr1[0][1])]
    patent1['distance']=arr1[0][0]
    patent2 = patentlist[patentlist['patent_num'] == str(arr1[1][1])]
    patent2['distance']=arr1[1][0]
    patent3 = patentlist[patentlist['patent_num'] == str(arr1[2][1])]
    patent3['distance']=arr1[2][0]
    patent4 = patentlist[patentlist['patent_num'] == str(arr1[3][1])]
    patent4['distance']=arr1[3][0]
    patent5 = patentlist[patentlist['patent_num'] == str(arr1[4][1])]
    patent5['distance']=arr1[4][0]
    
    bigdata1 = pd.concat([patent1, patent2,patent3,patent4,patent5], sort =False)
    bigdata=bigdata1[['patent_num','abstract','distance','title','url']]
    bigdata.to_csv('result_user.csv')
    return bigdata

def patentUserIdMatch(id):
    df= pd.read_excel('../data/UID_keyword_ptnum.xlsx')
    df =df.dropna()
    df
    user_list = df["UID"].tolist() 
    user_list
    if id in user_list:
        print("USERID is found")
        k = get_patentnumbers(id,df).values.tolist()
        res = get_similarity_matrix(k)
        return res
    else:
        print("USERID not there")

def getReadPatents(id):
    df= pd.read_excel('../data/UID_keyword_ptnum.xlsx')
    df =df.dropna()
    k = get_patentnumbers(id,df).values.tolist()
    patentno=k[0]
    patentlist = pd.read_csv('../data/Dataset.csv')
    df=pd.DataFrame()
    for i in range(len(patentno)):
        patent=patentlist[patentlist['patent_num'] == str(patentno[i])]
        df = df.append(pd.DataFrame(patent), ignore_index=True)
    return df[["patent_num","abstract","title","url"]]
    
