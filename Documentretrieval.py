
# coding: utf-8

# #Document retrieval from wikipedia data
# 
# #Import pandas

# In[1]:


import pandas as pd


# #Load some text data - from wikipedia, pages on people

# In[2]:


patentlist = pd.read_csv('Dataset.csv')


# Data contains:  link to wikipedia article, name of person, text of article.

# In[3]:


patentlist.head()


# In[4]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
cachedStopWords = set(stopwords.words("english"))
ps = PorterStemmer()


# In[5]:


len(patentlist)


# #Explore the dataset and checkout the text it contains
# 
# ##Exploring the entry for president Obama

# In[6]:


patent1 = patentlist[patentlist['patent_num'] == '10332155']


# In[7]:


patent1


# In[8]:


patent1['abstract'].values


# ##Exploring the entry for actor George Clooney

# In[9]:


patent2 = patentlist[patentlist['patent_num'] == '9642527']
patent2['abstract']


# #Get the word counts for Obama article

# In[10]:


from collections import Counter
word_count = []
count = Counter(str(patent1.abstract.values).split())
word_count.append(count)


# In[11]:


patent1['word_count'] = word_count
patent1.head()


# ##Sort the word counts for the Obama article

# ###Turning dictonary of word counts into a table

# In[12]:


patent1_word_count_table = pd.DataFrame(sorted(count.items(),key=lambda pair: pair[1],reverse=True))


# ###Sorting the word counts to show most common words at the top

# In[13]:


patent1_word_count_table.head()


# Most common words include uninformative words like "the", "in", "and",... Doesn't have much meaning!!

# #Compute TF-IDF for the corpus 
# 
# To give more weight to informative words, we weigh them by their TF-IDF scores.

# In[14]:


abstract=[]
for item in patentlist['abstract']:
    text=[]
    strtext=''
    for word in item.lower().split():
        word=word.replace(",", "").replace(".", "").replace("(", "").replace(")", "")
        if word not in cachedStopWords:
            word = ps.stem(word)
            text.append(word)
    strtext=' '.join(text)
    abstract.append(strtext)
patentlist['abstract']=abstract


# In[41]:


patentlist


# In[16]:


freq_count = []
for item in patentlist['abstract']:
    count = Counter(str(item).split())
    freq_count.append(count)
patentlist['word_count'] = freq_count
patentlist.head()


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


tfidfVect = TfidfVectorizer()
tfidf = tfidfVect.fit_transform(patentlist['abstract'])
tfidf


# In[19]:


print(tfidf)


# Count of occcurence of 'the' in the entire corpus

# In[40]:


print(tfidfVect.vocabulary_.get('radio'))


# ##Examine the TF-IDF for the Obama article

# In[21]:


patent1 = patentlist[patentlist['patent_num'] == '10332155']


# Find list of words in obama

# In[22]:


for words in patent1.word_count.items():
    print(set(words[1].elements()))
    patent1_words = set(words[1].elements())


# In[23]:


patent1


# Get Tfidf for Obama's text..
# fit with corpus and trasnform with obama's text

# In[24]:


patent1_tfidfVect = TfidfVectorizer()
patent1_tfidfVect = patent1_tfidfVect.fit(patentlist['abstract'])
patent1_tfidf = patent1_tfidfVect.transform(patent1['abstract'])


# In[25]:


patent1_tfidf.max()


# Vocabulary of the corpus with frequencies

# In[26]:


patent1_tfidfVect.vocabulary_
patent1_tfidf_table = pd.DataFrame(sorted(patent1_tfidfVect.vocabulary_.items(),key=lambda pair: pair[1],reverse=True))
patent1_tfidf_table


# In[27]:


print(patent1_tfidf)


# In[28]:


feature_names = patent1_tfidfVect.get_feature_names()
for col in patent1_tfidf.nonzero()[1]:
    print(feature_names[col], ' - ', patent1_tfidf[0, col])


# Get highest ranking words in obama text using TF IDF

# In[29]:


import numpy as np


# In[30]:


feature_array = np.array(feature_names)
tfidf_sorting = np.argsort(patent1_tfidf.toarray()).flatten()[::-1]
top_n = feature_array[tfidf_sorting][:10]
top_n


# Words with highest TF-IDF are much more informative.

# # #Build a nearest neighbor model for document retrieval
# 
# We now create a nearest-neighbors model and apply it to document retrieval.  

# In[31]:


#from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


# In[32]:


nbrs = NearestNeighbors(n_neighbors=10).fit(tfidf)
nbrs


# Find 10 nearest neighbours to Obama

# In[33]:


distances, indices = nbrs.kneighbors(patent1_tfidf)
distances,indices


# In[34]:


len(indices)


# In[35]:


names_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['patent_num'])
names_similar


# In[36]:


#abstract_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['abstract'])
#abstract_similar


# In[37]:


#url_similar = pd.Series(indices.flatten()).map(patentlist.reset_index()['url'])
#url_similar


# In[38]:


result = pd.DataFrame({'distance':distances.flatten(), 'patent_number':names_similar })
result=result.sort_values('distance')
result.to_csv('result.csv')
result[1:10]


# #Applying the nearest-neighbors model for retrieval

# ##Who is closest to Obama?

# As we can see, president Obama's article is closest to the one about his vice-president Biden, and those of other politicians.  
