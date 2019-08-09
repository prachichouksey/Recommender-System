# Patent Recommendation System

## Team member
Prachi Chouksey,
Pavani Somarouthu,
Ruchika Hazariwal and
Rachana Bumb

##Youtube link


## Overview
The proposed framework for building automatic recommendations in patents is composed of hybrid module: Content model and item-based collaborative model, and predict a recommendation list. The recommended objects are obtained by using a range of recommendation strategies based mainly on content based filtering and collaborative filtering approaches, each applied separately or in combination


## Data Sources
To analyze, manipulate, clean and evaluate our results on large amount of datasets, we have web scraped patents data from official USPTO website from where legal data scraping is done. Complete Web scraping was done using python’s Beautiful Soup module. Our dataset consists of approximately 13000 unique patent IDs with characteristics/features like patent number/ID, abstract, applicant city, applicant country, applicant name, applicant number, applicant state, applicant location, assignee name, cpc, family id, file date, inventors, patent date, title, URL. All these features/characteristics are being used to analyze the results for good recommendations by use of different keywords.
Link from where dataset is being web scraped: https://www.uspto.gov/

## Data Preprocessing
* Redundancy- We make sure redundant data is not present in our dataset while web scraping unique patents from the website.
* Missing Values- Few documents have missing abstracts, we fill in title in the abstract attribute to make sure data is complete.
* Attribute Selection- We have selected the following columns after preprocessing of data.
* Stemming- Generating variants of root/base words.

## Approaches
* User based Collaborative approach: This part of the module answers questions like: What is the distribution of how many patents a user interacts with in the dataset? Provides a visual and descriptive statistics to assist with giving a look at the number of times each user interacts with a patent
* Content based recommendation system: Patent classification is a problem in today’s, information and computer science. As a consequence of exponential growth, great importance has been put on the classification of patents into categories that describe the content of the abstract and body of patents. The function of classifier is to merge patent information into one or more predefined categories based on their content. Each patent can belong to several categories or may be its own category. Very rapid growth in the amount of text data leads to expansion of different automatic methods targeted to improve the speed and efficiency of automated patent classification and recommendation with textual content.
* Hybrid Approach: Content plus item based Collaborative recommendation system: Vector-based / Cosine-based Similarity: In this algorithm, patents are represented as two vectors that contain the user IDE and Patent ID. The similarity between user ID and Patent ID is calculated by the cosine of the angle between the two vectors. Database of approximately 20 users was created and stored as CSV file. Matrix of vectors is generated with rows and columns as User ID and Patent ID to know which user has read which patent of particular patent ID. Number represented in a row is matched to the Patent ID.

## Repo Structure
```
├── data
|   ├── Dataset.csv (Data fetched from USPTO)
|   ├── item_based_patents (Data about user)
|   └── similarity_matrix.csv (similarity matrix of every patent)
|
├── data-scraping
|   └── web_scrap.py (scraping functions for USPTO website)
|
├── web_app
|   ├── static (images, CSS and JavaScript files)
|   ├── templates (web-page templates)
|   ├── main.py (The main script file)
|   ├── app.py (the Flask application)
|   ├── tfidf_fit.pickle (pickle file)
|   ├── tfidf_fit_transform.pickle (pickle file)
|   └── tfidf_fit_transform.csv (tf-idf data)
```

## Deployment
The algorithms runs in batch mode, meaning that recommendations for each patent is calculated for every input. This flask application has been deployed on AWS EC2 instance,
