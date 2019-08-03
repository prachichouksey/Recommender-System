"""
Created on Tue Jul  23 17:02:27 2019

@author: prach
"""
from flask import Flask, render_template, request
import pandas as pd
import main

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', form_action="/recommender#search")

@app.route('/recommender', methods=['POST'])
def recommender():
    keyword = str(request.form['keyword'])
    searchCriteria = str(request.form['searchCriteria'])
#    try:
    if(searchCriteria=="keyword"):
        result=main.patentKeywordMatch(keyword)
    elif(searchCriteria=="userid"):
        result=main.patentUserIdMatch(keyword)
    elif(searchCriteria=="patentid"):
        result=main.patentPatentIdMatch(keyword)
    else:
        return render_template('index.html',
                       msg="Please select a valid criteria.",
                       form_action='/recommender#search')
    if not result.empty:
        if(searchCriteria=="patentid"):
            result_mypatent=result[:1]
            result=result[1:]
            return render_template('recommender.html', msg="", df=result,patent="true",df_mypatent=result_mypatent,form_action="/recommender#search")
        else:
            return render_template('recommender.html', msg="", df=result,form_action="/recommender#search")
    else:
        return render_template('index.html',
                               msg="Unfortunately, this patent is not in our database. Please try another paper.",
                               form_action='/recommender#search')

#    except:
#        return render_template('index.html',
#                           msg="Please enter valid search values.",
#                           form_action='/recommender#search')



# start the server with the 'run()' method
if __name__ == '__main__':
    app.run('0.0.0.0',5000,debug=True)