# Data Processing Component

## Description
* handles communication between web app and classification API
* applies preprocessing to the data before the classification (if the user selected this in the web app)

## Routes
* `fileupload`: the data from the web app is uploaded to this route. The upload is expected to be a dictionary containing the keys _model_ 
(string, the model-id of the selected model), _stopwords_ and _lowercase_ (booleans, whether the user selected these preprocessing options), 
_text_ and _date_ (strings, the uploaded data, different data points separated by `\n`).   
It returns the classifications results in a json file.

## Basic Usage
* activate the virtual environment
* run `flask run`

## Interface 
All files sent between the individual components are in the JSON format.
* the file sent from the web app to the processing component is expected to contain the following keys:
    * `text` (a single string containing all data points, separated by a newline character), 
    * `date` (also a single string of all dates, separated by a newline character), 
    * `model` (the id of the selected classifier as a string), 
    * `stopwords` (boolean whether or not stopwords should be removed), and 
    * `lowercase` (boolean whether or not stopwords should be removed).
* the file sent from the processing component to the classification API contains the following keys:
    * `model-id` (the id of the selected classifier as a string),
    * `text` (a list of the strings of text), and
    * `language` (a string containing the detected language of the texts).
* the file sent from the classification API to the processing component after sentiment analysis contains the following:
    * `probabilities` (a list of dicts, one for each classified sentence, containing the probabilities for `negative`, `neutral`, and `positive`), and
    * `sentiments` (a list of the dominant sentiment (as a string) for each sentence)
* the file sent from the classification API to the processing component after argument mining contains the following:
    * `probabilities` (a list of dicts, one for each classified sentence, containing the probabilities for `Argument_against`, `NoArgument`, and `Argument_for`), and
    * `argument_types` (a list of the most likely argument type for each sentence)
* the file sent from the processing component to the web app after sentiment analysis consists of one numbered dict per sentence, containing the following keys:
    * `text` (the original text),
    * `date` (the date of the text),
    * `language` (the detected language),
    * `preprocessed text` (the text with the selected preprocessing steps applied),
    * `dominant sentiment` (the detected sentiment of the text),
    * `neg` (the probability of the text being negative),
    * `neutral` (the probability of the text being neutral), and
    * `pos` (the probability of the text being positive)
* the file sent from the processing component to the web app after argument mining consists of one numbered dict per sentence, containing the following keys:
    * `text` (the original text),
    * `date` (the date of the text),
    * `language` (the detected language),
    * `preprocessed text` (the text with the selected preprocessing steps applied),
    * `Argument type` (the detected argument type of the text),
    * `against` (the probability of the text containing a contra argument),
    * `no argument` (the probability of the text containing no argument), and
    * `for` (the probability of the text containing a pro argument)
* the file sent from the classification API to the processing component after topic modeling contains the following keys:
    * `doc_lengths` (a list of the number of words in each data point),
    * `doc_topic_dists` (matrix of document-topic probabilities.),
    * `term_frequency` (a list of how often each topic is mentioned in the whole corpus),
    * `topic_term_dists` (matrix of topic-term probabilities in the size number of topics × len(vocab)), and
    * `vocab` (a list of all words in the corpus)   
This file is then forwarded to the web app, only the `language` (single string) is appended to it.
        