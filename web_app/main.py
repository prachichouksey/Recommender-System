import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.neighbors import NearestNeighbors

patentlist = pd.read_csv('../data/Dataset.csv')

patentlist.head()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
cachedStopWords = set(stopwords.words("english"))
ps = PorterStemmer()

from collections import Counter

def patentKeywordMatch(keyword):
    keyword_mod=[]
    keyword_mod.append(keyword)
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
    print(tfidf)
    print(tfidfVect.vocabulary_.get('radio'))
    patent1 =keyword_mod
    patent1_tfidfVect = TfidfVectorizer()
    patent1_tfidfVect = patent1_tfidfVect.fit(patentlist['abstract_org'])
    patent1_tfidf = patent1_tfidfVect.transform(patent1)
    patent1_tfidfVect.vocabulary_
#    patent1_tfidf_table = pd.DataFrame(sorted(patent1_tfidfVect.vocabulary_.items(),key=lambda pair: pair[1],reverse=True))
#    patent1_tfidf_table
#    feature_names = patent1_tfidfVect.get_feature_names()
#    for col in patent1_tfidf.nonzero()[1]:
#        print(feature_names[col], ' - ', patent1_tfidf[0, col])
    nbrs = NearestNeighbors(n_neighbors=6).fit(tfidf)
    distances, indices = nbrs.kneighbors(patent1_tfidf)
    names_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['patent_num'])
    abstract_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['abstract'])
    url_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['url'])
    title_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['title'])
    result = pd.DataFrame({'distance':distances.flatten(), 'patent_number':names_similar, 'abstract':abstract_similar, 'url':url_similar,
                      'title':title_similar})
    result=result.sort_values('distance')
    result.to_csv('result_keyword.csv')
    return result

def patentPatentIdMatch(keyword):
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
    print(tfidfVect.vocabulary_.get('radio'))
    patent1 = patentlist[patentlist['patent_num'] == keyword]
    patent1=patent1["abstract_org"]
    patent1_tfidfVect = TfidfVectorizer()
    patent1_tfidfVect = patent1_tfidfVect.fit(patentlist['abstract_org'])
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
    result = pd.DataFrame({'distance':distances.flatten(), 'patent_number':names_similar, 'abstract':abstract_similar, 'url':url_similar,
                      'title':title_similar})
    result=result.sort_values('distance')
    result.to_csv('result_patentid.csv')
    
    return result
    
if __name__=="__main__":
    test=patentKeywordMatch("cloud")
    print(test)
