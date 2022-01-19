import streamlit as st
import requests
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
# for lda vis
import pyLDAvis
import streamlit.components.v1 as components
import json
import helper_functions as helper
import visualisation_functions as vis
#
from typing import Dict

# global variables
processing_component_url = 'http://127.0.0.1:8000/fileupload'
# deactivate deprecation warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# persistent dataframe that is being cached throughout application as long as dataset stays the same
# initialize it as empty dataframe that is filled later
# see _get_cached_df
df = pd.DataFrame([])

# ___________________________ define helper functions here ____________________

# Posts data and selected options to data processing component
@st.cache(suppress_st_warning=True)
def _upload_to_processing(df, classifier):
    texts = ''
    for line in df['full_text']:
        texts += line + '\n'

    dates = ''
    for line in df['created_at']:
        dates += line + '\n'

    data_obj = {'text': texts, 'date': dates, 'model': classifier,
                'stopwords': stopwords_check, 'corpus specific': corpus_specific_check,
                'lowercase': lowercase_check}

    return requests.post(processing_component_url, data=data_obj)


# ___________________________ layout code starts here _______________________

st.title('Visualising the Evolution of Debates')

# __________________________ sidebar code ___________________________________

models = helper._get_available_models()
model_dict = {'Sentiment Analysis':'sa-distilbert', 'Argument Mining':'am-distilbert', 'Topic Modelling': 'tm-lda' }
analysis_type = st.sidebar.selectbox('Choose an analysis type ', list(model_dict.keys()))
classifier = model_dict[analysis_type]
uploaded_file = st.sidebar.file_uploader('Upload your data', ['tsv'])
meta_data_check = st.sidebar.checkbox('Show classifier metadata?', value=False)
if meta_data_check:
    hyperparameters, performance, training = helper._get_meta_info(classifier)
    if (hyperparameters):
        st.markdown(hyperparameters)
        st.markdown('---')
    if (performance):
        st.markdown(performance)
        st.markdown('---')
    if (training):
        st.markdown(training)
        st.markdown('---')

st.sidebar.markdown('**Choose the preprocessing options for your data:**')
stopwords_check = st.sidebar.checkbox('Remove regular stopwords')
corpus_specific_check = st.sidebar.checkbox('Remove corpus-specific stopwords')
lowercase_check = st.sidebar.checkbox('Make all letters lowercase')

st.sidebar.markdown('**Apply filters**')
keywords_input = st.sidebar.text_area('''Enter the keywords you want to filter for, separated by a new line. Press ctrl+enter to apply the filter.
                                        To delete a keyword, delete it from the list and press ctrl+enter again.''')
keywords_list = keywords_input.split('\n')
keywords_reg = r'(?i){}'.format('|'.join(keywords_list))
if keywords_list[0] is not '':
    keywords_list_emp = ['*{}*'.format(keyword) for keyword in keywords_list]
    keywords_str = ', '.join(keywords_list_emp)
    st.info('You are currently applying filters : ' + keywords_str)
st.sidebar.markdown('**Apply time filter**')
time_filter = st.sidebar.date_input("This is a time filter, please select start date and end date:", [])

# actually not strictly separated. l.120 to 147 is layout-code
# _________________________ main page code________________________________

# Executed if user has selected a file
if (uploaded_file):

    with st.spinner('Processing...'):
        # Executed after result is received
        df = pd.read_csv(uploaded_file, sep='\t')
        # get the cached dataframe when the same dataset is used
        # then mutate cached dataframe. This makes use of streamlits caching funtionality.
        cached_df = helper._get_cached_df(df)
        arguments_computed = helper._are_arguments_computed(df)
        sentiments_computed = helper._are_sentiments_computed(df)
        if 'tm' in classifier:
            # handle user filter input
            st.sidebar.markdown('** Search by Arguments**')
            # check whether arguments or sentiments were computed already
            if not arguments_computed['computed'] or not sentiments_computed['computed']:
                st.sidebar.markdown(
                    'In order to search by arguments or sentiments, please **execute** the respective model in advance')
            argument_filter = st.sidebar.selectbox(
                'Enter the argument type you want to filter for. Only those arguments are '
                'utilized for visualisation.',
                options=['No Filter', 'Pro', 'No Argument', 'Contra'])
            st.sidebar.markdown('** Search by Sentiments**')
            sentiment_filter = st.sidebar.selectbox(
                'Enter the sentiment you want to filter for. Only those arguments are '
                'utilized for visualisation.',
                options=['No Filter', 'Positive', 'Neutral', 'Negative'])
            # If a filter was chosen, but sentiments or arguments were not computed before
            if sentiment_filter is not 'No Filter' and not sentiments_computed['computed']:
                st.warning(
                    'Sentiment Analysis Model has to be executed before filtering by sentiments. Filter is set to *No Filter*')
                sentiment_filter = 'No Filter'
            else:
                if sentiment_filter is not 'No Filter':
                    st.info(
                        'You are currently applying argument filter *{}*'.format(sentiment_filter))
            if argument_filter is not 'No Filter' and not arguments_computed['computed']:
                st.warning(
                    'Argument Mining Model has to be executed before filtering by arguments. Filter is set to *No Filter*')
                argument_filter = 'No Filter'
            else:
                if argument_filter is not 'No Filter':
                    st.info('You are currently applying argument filter *{}*'.format(
                        argument_filter))
            sentiment_keys = {'Positive': 'positive', 'Neutral': 'neutral', 'Negative': 'negative'}
            argument_keys = {'Pro': 'Argument_for', 'No Argument': 'NoArgument', 'Contra': 'Argument_against'}
            sentiment_indices = pd.Series([True] * len(cached_df.index))
            if sentiment_filter is not 'No Filter':
                sentiment_indices = cached_df['dominant sentiment'] == sentiment_keys[sentiment_filter]
            argument_indices = pd.Series([True] * len(cached_df.index))
            if argument_filter is not 'No Filter':
                argument_indices = cached_df['Argument type'] == argument_keys[argument_filter]
            # apply argument and sentiment filter
            df = df[argument_indices & sentiment_indices]
            # apply keyword filter
            df = df[df['full_text'].str.contains(keywords_reg)]
            if sentiment_filter is not 'No Filter' or argument_filter is not 'No Filter':
                st.sidebar.info(f'Remaining data after filtering:  {len(df.index)}')
        if df.empty:
            st.warning('No data points fulfill filtering criteria')
        else:
            # Show data-related information
            st.markdown('** Data Info**')
            text = helper._document_info(df, 'english')
            start_time, end_time = helper._document_time_range(df['created_at'])
            st.markdown('** Word Cloud **')
            st.write('This is a wordcloud representation of the most used words in your data')
            helper._show_wordcloud(text)
            time_period = end_time - start_time
            resp = _upload_to_processing(df, classifier)
            resp_dict = resp.json()
            result = pd.DataFrame.from_dict(resp_dict, orient='index')
            response_dict = json.loads(resp.text)

            if 'am' in classifier or 'sa' in classifier:
                if (result['language'][0] != 'english'):
                    st.warning(
                        'It appears that the data you uploaded is not in English. Currently, there are only classifiers for English-language texts. Please be aware that the following results may not be accurate.')
                    # if using am or sa cache the predictions to cached_df for later use in tm
                if 'sa' in classifier and 'dominant sentiment' not in cached_df.columns.values.tolist():
                    for key in ['dominant sentiment', 'neg', 'neutral', 'pos']:
                        if key in result:
                            cached_df[key] = result[key].to_list()
                    sentiments_computed['computed'] = True
                elif 'am' in classifier and 'Argument type' not in cached_df.columns.values.tolist():
                    for key in ['Argument type', 'against', 'no argument', 'for']:
                        if key in result:
                            cached_df[key] = result[key].to_list()
                    arguments_computed['computed'] = True

                helper._append_keywords(result, keywords_list)
                st.markdown('**Activity Chart**')
                activity_chart = vis._activity_chart(df)
                st.altair_chart(activity_chart, use_container_width=True)

                if (time_filter!=()):  # not same day
                    if (len(time_filter)==1):
                        result=helper._time_filter(result,time_filter)
                        start_time=pd.to_datetime(time_filter[0],utc=False)
                    else:
                        result=helper._time_filter(result,time_filter)
                        start_time=pd.to_datetime(result.date.min(),utc=False)
                        end_time=pd.to_datetime(result.date.max(),utc=False)

                if result.empty:  #no data chosen
                    st.write('No data in this time range!')

                df_vis = vis._prepare_result_for_line(result, classifier)
                sent_arg_to_num_dict = {'positive': 1, 'neutral': 0, 'negative': -1, 'Argument_for': 1, 'NoArgument': 0,
                                        'Argument_against': -1}

                if (classifier == 'sa-distilbert'):
                    x = pd.to_datetime(df_vis['date'])
                    y = np.array([sent_arg_to_num_dict[sent] for sent in df_vis['dominant sentiment']])
                    vis_text = np.array(df_vis['text'])
                    vis_title = 'Sentiment Line Chart'
                    y_title = 'Dominant Sentiment'
                    ticktext = ['positive', 'neutral', 'negative']
                else:
                    x = pd.to_datetime(df_vis['date'])
                    y = np.array([sent_arg_to_num_dict[sent] for sent in df_vis['Argument type']])
                    vis_text = np.array(df_vis['text'])
                    vis_title = 'Argument Line Chart'
                    y_title = 'Argument Type'
                    ticktext = ['Argument for', 'No argument', 'Argument against']

                new_vis_text = []

                # Adds line breaks (if necessary) to the tweets that are displayed when hovering over datapoints
                for text in vis_text:
                    if (len(text) > 120):
                        for i in range (len(text)):
                            if ((i % 120) > 100 and text[i] == ' ' and not '<br>' in text[i-20:i]):
                                text = text[:i] + '<br>' + text[i+1:]

                    new_vis_text.append(text)

                scatter_fig = go.Figure()
                scatter_fig.add_trace(go.Scatter(x=x, y=y, text=new_vis_text, hoverinfo='text', hoverlabel = dict(namelength = -1), mode='markers', name=y_title))
                scatter_fig.update_yaxes(nticks=3, tickvals=[1, 0, -1], ticktext=ticktext)

                if (st.checkbox('Show trend', value=True) and (not result.empty)):  # adds the over all trendline
                    fig = px.scatter(df_vis, x=x, y=y, trendline='lowess', trendline_color_override='red')
                    trendline = fig.data[1]
                    scatter_fig.add_trace(go.Scatter(trendline, name='Overall Trend', showlegend=True))
                    scatter_fig.update_layout(showlegend=True)

                # this list is iterated through to ensure different colours for the different
                list_of_colors = ['green', 'yellow', 'blue', 'purple', 'black', 'orange', 'coral', 'cyan', 'beige',
                                  'chocolate']
                color_i = 0
                sent_or_arg = 'dominant sentiment' if 'sa' in classifier else 'Argument type'
                for keyword in keywords_list:
                    if (keyword):
                        keyword_df = df_vis[df_vis['text'].str.contains(keyword)]
                        if (not keyword_df.empty):
                            x = pd.to_datetime(keyword_df['date'])
                            y = np.array([sent_arg_to_num_dict[sent] for sent in keyword_df[sent_or_arg]])

                            fig = px.scatter(keyword_df, x=x, y=y, trendline='lowess',
                                             trendline_color_override=list_of_colors[color_i])
                            trendline = fig.data[1]
                            description = 'Texts containing {0}'.format(keyword)
                            scatter_fig.add_trace(go.Scatter(trendline, name=description, showlegend=True))
                            scatter_fig.update_layout(showlegend=True)
                            color_i = (color_i + 1) % 10


                # filters out the datapoints not matching any of the keywords
                result = result[result['text'].str.contains(keywords_reg)]
                if keywords_list[0] is not '':
                    st.sidebar.info(f'Remaining data after filtering:  {len(result.index)} ')
                if (not result.empty):
                    scatter_fig.update_layout(title=vis_title, xaxis_title='Date', yaxis_title=y_title)
                    st.plotly_chart(scatter_fig, use_container_width=False)

                    st.markdown('---')

                    time_interval = helper._time_interval(end_time - start_time,'bar')
                    st.markdown('Please select the **appropriate** time interval according to your dataset.')
                    if (classifier == 'sa-distilbert'):
                        res = vis._prepare_result_for_bar(result, time_interval, 'sentiment')
                        bar_fig = vis._calculate_bar_area_fig(res, 'sentiment', 'Sentiment Bar Chart', time_interval, 'bar')
                    else:
                        res = vis._prepare_result_for_bar(result, time_interval, 'argument')
                        bar_fig = vis._calculate_bar_area_fig(res, 'argument', 'Argument Bar Chart', time_interval, 'bar')

                    st.plotly_chart(bar_fig, use_container_width=False)

                    time_interval_area = helper._time_interval(end_time - start_time,'area')
                    st.markdown('Please select the **appropriate** time interval according to your dataset.')
                    st.markdown('$$Attention!$$ If the X-axis of the time interval you choose has only one value, no chart will be generated.')
                    if (classifier == 'sa-distilbert'):
                        res = vis._prepare_result_for_area(result, time_interval_area, 'sentiment')
                        area_fig = vis._calculate_bar_area_fig(res, 'sentiment', 'Sentiment Area Chart', time_interval_area,
                                                               'area')
                    else:
                        res = vis._prepare_result_for_area(result, time_interval_area, 'argument')
                        area_fig = vis._calculate_bar_area_fig(res, 'argument', 'Argument Area Chart', time_interval_area,
                                                               'area')
                    st.plotly_chart(area_fig, use_container_width=False)
                else:
                    st.write('No data points match your filters. Please choose different filters.')
            if (classifier == 'tm-lda'):
                if (response_dict['language'] != 'english'):
                    st.warning(
                        'It appears that the data you uploaded is not in English. Currently, there are only classifiers for English-language texts. Please be aware that the following results may not be accurate.')
                st.markdown('**Topic Model Visualization**')
                st.markdown('This visualization is based on the work from [Sievert and Shirley](nlp.stanford.edu/events/'
                            'illvi2014/papers/sievert-illvi2014.pdf).'
                            ' Following is an explanation of the visual elements in the graph : \n'
                            '* **Circles** represent a topic, whose area '
                            'is proportional to the proportion of the topic in the corpus. The distance between the circles '
                            'describe the degree of similarity between topics.\n '
                            '* **Red bars** represent the number of times a term was generated by the specified topic. \n'
                            '* **Blue bars** represent the overall frequency of terms in the corpus. \n'
                            '* **Relevance metric $\lambda$ ** controls the relevance of terms specific to the topic. The lower $\lambda$ is the more specific are the terms.')
                response_dict = json.loads(resp.text)
                response_dict.pop('language')
                vis_data = pyLDAvis.prepare(**response_dict)
                vis_html = pyLDAvis.prepared_data_to_html(vis_data)
                components.html(vis_html, height=1000, width=2500)
else:
    st.info("* **Please use the sidebar to upload your data.**  \n * The **accepted format** is a tsv file with text in a column named *'full_text'* and the corresponding date in a column named *'created_at'*.")
    st.markdown("Welcome to our Visualisation Application. With this app you can upload your own data of a debate and "
                "visualise its evolution in terms of its *sentiments*, *argument-types*, and *topics*. Below you can "
                "find a description of the functionalities of our app : \n")
    st.markdown("* **Sentiment Analysis** : Uploaded sentences will be classified as positive, negative or neutral "
                "sentiments. You can see how the sentiments of the debate evolve over time in the different plots "
                "shown. \n")

    st.markdown("* **Argument Mining** : Uploaded sentences will be classified as pro-arguments, contra-arguments or non-arguments. \n" )
    st.markdown("* **Topic Modelling** : Using *LDA* we visualize the topics that occur within the debate. \n"
                "Here a topic is comprised of a list of words that describe this topic. \n")
    st.markdown("* **Keyword Filtering** : You can filter your uploaded debate with respect to the keywords that should appear"
                "in the sentences. Here the filter is *inclusive*, that is all keyword filters are applied as inclusive or-operations. \n")
    st.markdown("* **Time Filerting**: If you want restrict the visualisation of the debate to a particular time frame, "
                "you can use this tool. \n")
    st.markdown("* **Argument - and Sentiment Filtering** : Additionally, when using the Topic Modelling visualisations, you can visualise "
                "topics that appear for particular argument or sentiment types. In order to use this filer, you have to execute the corresponding "
                "analysis type for the uploaded dataset beforehand. For example to filter by arguments, you need to execute Argument Mining first. ")