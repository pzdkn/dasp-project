from flask import render_template, request, flash, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from app import app
import requests
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer

api_endpoint = 'http://127.0.0.1:5005/predict' 
selected_model = 'sa-distilbert' # default value to declare global variable

CORS(app)
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')


# The web app sends the uploaded file to this route.
@app.route('/fileupload', methods=['POST'])
def fileupload():
    selected_model = request.form['model']
    remove_stopwords = request.form['stopwords']
    remove_corpus_specific = request.form['corpus specific']
    lowercase_words = request.form['lowercase']
    
    full_text = request.form['text'].splitlines()
    created_at = request.form['date'].splitlines()
    df = pd.DataFrame(list(zip(full_text, created_at)), columns=['text', 'date'])

    # The language is needed to remove the correct stopwords. If the language cannot be determined,
    # no stopwords are removed. The user receives a warning in the web app if the language of their 
    # data is not supported by a classifier. 
    language = _define_language(full_text)
    df['language'] = language

    if (remove_stopwords == "True" and language != 'unknown'):
        stop_words = set(stopwords.words(language))
        df['preprocessed text'] = [_remove_stopwords(text, stop_words) for text in df['text']] 
    else:
        df['preprocessed text'] = df['text']

    if (remove_corpus_specific == "True"):
        preprocessed_as_list = df['preprocessed text'].to_list()
        vectorizer = CountVectorizer(preprocessed_as_list, max_df = 0.5) # Words that appear in more than 50% of texts are treated as stopwords
        X = vectorizer.fit_transform(preprocessed_as_list)
        additional_stopwords = vectorizer.stop_words_
        df['preprocessed text'] = [_remove_stopwords(text, additional_stopwords) for text in df['preprocessed text']]

    if(lowercase_words == "True"):
        df['preprocessed text'] = [text.lower() for text in df['preprocessed text']]

    
    if (selected_model == 'sa-distilbert'):
        _classify_sentiment(df, selected_model, language)
        result = df.to_json(orient='index')

    elif (selected_model == 'tm-lda'):
        result = _classify_topic(df, selected_model, language)

    elif (selected_model == 'am-distilbert'):
        _classify_argument(df, selected_model, language)
        result = df.to_json(orient='index')
        
    return result


# This sends the text data to the sentiment classifier and adds the result in a new column to the dataframe.
# Input: dataframe containing at least column 'text' and URL of selected classifier
def _classify_sentiment(df, model_id, language):
    sentences = [sentence for sentence in df['preprocessed text']]

    dataDict = {
        'model-id': model_id, 
        'text': sentences,
        'language': language
    }
    r = requests.post(api_endpoint, json = dataDict)
    response = r.json()

    df['dominant sentiment'] = response['sentiments']
    df['neg'] = [response_dict['negative'] for response_dict in response['probabilities']]
    df['neutral'] = [response_dict['neutral'] for response_dict in response['probabilities']]
    df['pos'] = [response_dict['positive'] for response_dict in response['probabilities']]


# This sends the text data to the topic classifier and returns a json of the results.
# Input: dataframe containing at least column 'text' and URL of selected classifier
def _classify_topic(df, model_id, language):
    sentences = [sentence for sentence in df['preprocessed text']]

    dataDict = {
        'model-id': model_id, 
        'text': sentences,
        'language': language
    }
    r = requests.post(api_endpoint, json = dataDict)
    response = r.json()
    response.update({'language': language})
    return response


# This sends the text data to the argument classifier and adds the result in a new column to the dataframe.
# Input: dataframe containing at least column 'text' and URL of selected classifier
def _classify_argument(df, model_id, language):
    sentences = [sentence for sentence in df['preprocessed text']]

    dataDict = {
        'model-id': model_id, 
        'text': sentences,
        'language': language
    }

    r = requests.post(api_endpoint, json = dataDict)
    response = r.json()
    
    df['Argument type'] = response['argument_types']
    df['against'] = [response_dict['Argument_against'] for response_dict in response['probabilities']]
    df['no argument'] = [response_dict['NoArgument'] for response_dict in response['probabilities']]
    df['for'] = [response_dict['Argument_for'] for response_dict in response['probabilities']]


# Uses the langdetect package to detect the language of the data.
# If two out of the three first texts are detected as the same language, this language is selected.
def _define_language(data):
    if (len(data) >= 3):
        lang_list = [detect(text) for text in data[:3]]
        if (lang_list.count(lang_list[0]) >= 2):
            lang_guess = lang_list[0]

    else:
        lang_guess = detect(data[0])

    if (lang_guess == 'en'):
        return 'english'
    
    elif (lang_guess == 'de'):
        return 'german'

    elif (lang_guess == 'es'):
        return 'spanish'

    elif (lang_guess == 'fr'):
        return 'french'
        
    else:
        return 'unknown'


# Uses the NLTK stopwords corpus to remove stopwords.
# Input: a string and a list of stopwords that are to be removed
# Output: the same string with all stopwords removed
def _remove_stopwords(text, stopwords):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stopwords]
    separator = ' '
    return separator.join(filtered_sentence)