#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def get_patentnumbers(id,df):
    subsetDataFrame = df['PATENT NUMBER'][df['UID'] == id].str.split(",", expand = True) 
    return subsetDataFrame


def get_similarity_matrix(k):
    numbers = k[0]
    for i in range(0, len(numbers)): 
        numbers[i] = int(numbers[i])
    len(numbers)
    data= pd.read_csv('similarity_matrix.csv')
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
    
    patentlist = pd.read_csv('Dataset.csv')
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

def search_Userid(id):
    df= pd.read_excel('UID_keyword_ptnum.xlsx')
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


# In[3]:

if __name__ == '__main__':
    id='U02'
    search_Userid(id)


# In[ ]:




