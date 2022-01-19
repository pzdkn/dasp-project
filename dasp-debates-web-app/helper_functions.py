import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from datetime import timedelta


# global variables
meta_info_url = 'http://127.0.0.1:5005/meta_info'
available_models_url = 'http://127.0.0.1:5005/available_models'



#___________________________ define helper functions here ____________________
def hash_func(df: pd.DataFrame):
    '''
    custom hashing function. See [https://docs.streamlit.io/en/stable/advanced_caching.html#the-hash-funcs-parameter]
    checks for the first 10 sentences of the dataframe, if it's the same use the cache
    '''
    if df.empty:
        return None
    elif 'full_text' in df:
        return df['full_text'][:10]
    else:
        return df['text'][:10]


@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: hash_func})
def _get_cached_df(df):
    # get a cached version of the dataframe. It will retrieve from the cache if
    # the first 10 sentences are the same
    return df

@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: hash_func})
def _are_sentiments_computed(df):
    # a cached value to indicate whether sentiments have being computed for that specific df ?
    return {'computed': False}

@st.cache(allow_output_mutation=True, hash_funcs={pd.DataFrame: hash_func})
def _are_arguments_computed(df):
    return {'computed': False}

# Uses the NLTK stopwords corpus to remove stopwords.
# Input: a string and the language of the string (as a lowercase string, e.g. 'english')
# Output: the same string with all stopwords removed
def _remove_stopwords(text, language):
    stop_words = set(stopwords.words(language))
    additional_stopwords = ['rt', 'https']
    for word in additional_stopwords:
        stop_words.add(word)
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    separator = " "
    return separator.join(filtered_sentence)


# Displays the timeframe of the dataset
def _document_time_range(time_df):
    start_time=pd.to_datetime(time_df.min(),utc=False)
    end_time=pd.to_datetime(time_df.max(),utc=False)
    st.write('* The time range of the data you uploaded starts from ', start_time, 'to', end_time, '\n')
    return start_time, end_time


# Displays basic information about the file while the data is being classified.
# Currently shows the number of datapoints in the uploaded file and a wordcloud.
def _document_info(df, language):  
    text = ''
    for line in df['full_text']: 
        text += line

    number_of_data = len(df['full_text'])
    st.write('* Uploaded data contains ', number_of_data, 'data points. \n')

    text = _remove_stopwords(text, 'english')
    return text


def _show_wordcloud(text):
    wordcloud = WordCloud().generate(text)
    # st.write('This is a wordcloud representation of the most used words in your data:')
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()

def _time_interval(time_period,chart):
    if (time_period > timedelta(days=365)):
        m=["year","month"]
    elif (time_period > timedelta(days=30)):    
        m=["month","date"]
    elif (time_period > timedelta(hours=1)):    
        m=["date","hours"]
    elif (time_period > timedelta(minutes=1)):  
        m=["hours","minutes"]
    elif (time_period > timedelta(seconds=1)):
        m=["minutes","seconds"]
    else:  #only one node
        m=["seconds"]
    if (chart=="bar"):
        time_interval = st.selectbox('We recommend that you use the following time intervals to generate bar chart visualisations:',m)
    elif (chart=="area"):
        time_interval = st.selectbox('We recommend that you use the following time intervals to generate area chart visualisations:',m)
    return time_interval


# Returns a json file containing meta data of the specified classifier
def _get_meta_info(classifier):
    data = {'model-id': classifier}
    r = requests.post(meta_info_url, json = data)
    rjson = r.json()
    
    hyperparameters = rjson['hyperparameters']
    hyperparameter_string = ""
    if hyperparameters is not None:
        hyperparameter_string += "***Hyperparameters: ***  \n"
        for (key, value) in hyperparameters.items():
            hyperparameter_string += f"**{key}**: {value}  \n"

    performance = rjson['performance']
    performance_string = ""
    if performance is not None:
        performance_string += "***Performance: ***  \n"
        for (key, value) in performance.items():
            performance_string += f"**{key}**: {value}  \n"

    parameters = rjson['training-parameters']
    training_string = ""
    if parameters is not None:
        training_string += "***Training Parameters: ***  \n"
        for (key, value) in parameters.items():
            training_string += f"**{key}**: {value}  \n"

    return (hyperparameter_string, performance_string, training_string)



# Returns a list of all available classification models     
def _get_available_models():
    r = requests.get(available_models_url)
    models = r.json()['available-models']
    return models


def _append_keywords(df, keywords):
    for keyword in keywords:
        df[keyword] = [_contains(text, keyword) for text in df['text']]

def _contains(text, word):
    if (word in text): 
        return True
    else:
        return False

def _time_filter(df,time_filter):
    start_date=str(time_filter[0])
    df = df[df.date.str[0:10] >= start_date]
    if len(time_filter)>1:
        end_date=str(time_filter[1])
        df = df[df.date.str[0:10] <= end_date]
    return df