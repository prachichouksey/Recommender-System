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

    try:
        result=main.patentKeywordMatch(keyword)
        print(result)
        if not result.empty:
            return render_template('recommender.html', msg="", df=result,form_action="/recommender#search")
        else:
            return render_template('index.html',
                                   msg="Unfortunately, this paper is not in our database. Please try another paper.",
                                   form_action='/recommender#search')

    except:
        return render_template('index.html',
                           msg="Please enter a valid paper id.",
                           form_action='/recommender#search')



# start the server with the 'run()' method
if __name__ == '__main__':
    app.run('0.0.0.0',5000,debug=True)