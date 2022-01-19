import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import calendar
from datetime import datetime,timedelta

import helper_functions as helper


def _extract_time(df,time_interval):
    res = pd.DataFrame(columns=['date'])
    if time_interval=="seconds":
        res['date']=df['date']
    elif time_interval=="minutes":
        res['date']=df['date'].str[0:16]  
    if time_interval=="hours":
        res['date']=df['date'].str[0:13]   
    elif time_interval=="date":
        res['date']=df['date'].str[0:10]
    elif time_interval=="month":
        res['date']=df['date'].str[0:7]
    elif time_interval=="year":
        res['date']=df['date'].str[0:4]+'-01-01 00:00:00'
    return res['date']

# Prepares the dataframe into the shape necessary for the bar chart visualisation for the sentiment model.
@st.cache(allow_output_mutation=True)  # cache result because it takes a long time
def _prepare_result_for_bar(df, time_interval, data_kind):
    df = df.sort_values(by = 'date')
    res = pd.DataFrame(columns=['date', data_kind])
    res['date'] = _extract_time(df,time_interval)
    if (data_kind == 'sentiment'):
        res["sentiment"]=df["dominant sentiment"]
        res_01=res[['date',data_kind]].groupby(['date',data_kind]).sentiment.agg('count').to_frame('count').reset_index()
    elif (data_kind == 'argument'):
        res["argument"]=df["Argument type"]
        res_01=res[['date','argument']].groupby(['date','argument']).argument.agg('count').to_frame('count').reset_index()
    return res_01


@st.cache(allow_output_mutation=True) # cache result because it takes a long time
def _prepare_result_for_area(df, time_interval, data_kind):
    df = df.sort_values(by = 'date')
    res = pd.DataFrame(columns=['date', data_kind]) 
    res['date'] = _extract_time(df,time_interval)
    if (data_kind == 'sentiment'):
        res[data_kind] = df['dominant sentiment']
        res_01=res[['date',data_kind]].groupby(['date',data_kind]).sentiment.agg('count').to_frame('count').reset_index()
    elif (data_kind == 'argument'):
        res[data_kind] = df['Argument type']
        res_01=res[['date',data_kind]].groupby(['date',data_kind]).argument.agg('count').to_frame('count').reset_index()
    a = list(res_01['date'].unique())   # The unique value of the column
    res_03 = pd.DataFrame(columns=['date', data_kind,'count','percentage']) 
    for i in range(len(a)):
        if (data_kind == 'sentiment'):
            res_02 = pd.DataFrame({'date': [a[i],a[i],a[i]], data_kind: ['negative','neutral','positive'],'count': [0, 0,0]},index=['negative','neutral','positive'])
        elif (data_kind == 'argument'):
            res_02 = pd.DataFrame({'date': [a[i],a[i],a[i]],'argument': ['Argument_against','Argument_for','NoArgument'],'count': [0,0,0]},index=['Argument_against','Argument_for','NoArgument'])  # a empty dataframe
        b = res_01[(res_01.date==a[i])].set_index([data_kind])
        res_03 = res_03.append(b.combine_first(res_02))
    print(res_03)
    for i in range(int(len(res_03)/3)):
        _sum=res_03['count'][i*3]+res_03['count'][i*3+1]+res_03['count'][i*3+2]
        res_03['percentage'][i*3]=res_03['count'][i*3]/_sum*100
        res_03['percentage'][i*3+1]=res_03['count'][i*3+1]/_sum*100
        res_03['percentage'][i*3+2]=res_03['count'][i*3+2]/_sum*100
    print(res_03)
    # for i in range(len(res_03)-3):     #cummulative
    #     res_03['count'][i+3]=res_03['count'][i]+res_03['count'][i+3] 
    return res_03


# Calculates the bar chart and area chart for the visualisation.
def _calculate_bar_area_fig(res, color, title,  time_interval, kind_of_vis):
    if (kind_of_vis == 'bar'):
        fig = px.bar(res, x="date", y="count", color=color, title=title)
    elif (kind_of_vis == 'area'):
        fig = px.area(res, x="date", y="percentage", color=color, title=title)
        fig.layout.yaxis.ticksuffix='%'
    print(res['date'])
    if time_interval=="minutes":
        fig.layout.xaxis.tickvals = pd.date_range(res['date'].min(), res['date'].max(), freq='min')
        fig.layout.xaxis.tickformat = "%d-%b-%Y %H:%M m"
    elif time_interval=="hours":
        fig.layout.xaxis.tickvals = pd.date_range(res['date'].min(), res['date'].max(), freq='H')
        fig.layout.xaxis.tickformat = "%d-%b-%Y %H h"
    elif time_interval=="date":
        fig.layout.xaxis.tickvals = pd.date_range(res['date'].min(), res['date'].max(), freq='D')
        fig.layout.xaxis.tickformat = "%d-%b-%Y"
    elif time_interval=="month":
        fig.layout.xaxis.tickvals = pd.date_range(res['date'].min(), res['date'].max(), freq='MS')
        fig.layout.xaxis.nticks=20
        fig.layout.xaxis.tickformat = "%b-%Y"
    elif time_interval=="year": 
        for i in range(len(res)):
            res['date'][i]=datetime.strptime(str(res['date'][i]),"%Y-%m-%d %H:%M:%S")
        fig.layout.xaxis.tickvals = pd.date_range(res['date'].min(), res['date'].max(), freq='YS')
        fig.layout.xaxis.tickformat = "%Y"
    
    fig.update_layout(xaxis = dict(tickmode = 'auto',dtick = 3))
    return fig



# Prepares the dataframe into the shape necessary for the line chart visualisation.
@st.cache # cache result because it takes a long time
def _prepare_result_for_line(df, classifier):
    # df['date'] = pd.to_datetime(df['date'], utc=True) 
    df = df.sort_values(by = 'date')
    if (classifier == 'sa-distilbert'):
        df = df.drop(['preprocessed text', 'neg', 'pos', 'neutral'], axis = 1) # Not needed for current visualisations, maybe use later?
    if (classifier == 'am-distilbert'):
        df = df.drop(['preprocessed text', 'against', 'no argument', 'for'], axis = 1)
    return df

def _activity_chart(df):
    st.write('The number of sentences in different months:')
    df = df.sort_values(by = 'created_at')
    df['Month']=df['created_at'].str[5:7]
    df['Date']=df['created_at'].str[0:4]
    print(df)
    for i in range(len(df)):
        df['Date'][i]=datetime.strptime(df['Date'][i],"%Y")
    for i in range(len(df)):
        df['Month'][i]=calendar.month_abbr[int(df['Month'][i])]
    res=df[['Date','Month']].groupby(['Date','Month']).Month.agg('count').to_frame('count').reset_index()
    print(res)
    c = alt.Chart(res).mark_circle().encode(
            alt.X('Date',axis=alt.Axis(format='%Y'),title='Year'),
            alt.Y('Month',scale=alt.Scale(domain=('Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))),
            size='count', color='count', tooltip=['year(Date)', 'Month', 'count']).configure_axis(grid=True)
    return c