# Web App

## Description
* User can upload his own data set and choose a classifier. The classification results are shown in appropriate visualisations.
* The user can also choose if pre-processing should be applied to the data
* It is possible to filter the classification results by keywords, or by sentiment or argument that the classifier discovered and see how the visualisations change.

## Basic Usage
* Activate the virtual environment
* run `streamlit run evolution_of_debates.py`

## File Structure   
There are three main files:   
* `evolution_of_debates.py` contains the layout code for the web app. 
* `helper_functions.py` contains outsourced functions needed for the web app.
* `visualisation_functions.py` contains the code for the visualisations.

## Add new visualisation
We used [streamlit](https://www.streamlit.io/) for the web app, which supports many visualisation libraries 
(see the [documentation](https://docs.streamlit.io/en/stable/api.html#display-charts)).
Since we used [plotly](https://plotly.com/python/) for most of the visualisations, here's a guide on how to include a new plotly chart:   
* create the figure just as you would in plotly (see [here](https://plotly.com/python/renderers/) for more infos)
* instead of `fig.show`, use `st.plotly_chart(fig)`
* upon saving the code, the streamlit app should ask for permission to restart in the top right corner of the browser
* click _Rerun_ or _Always Rerun_ to restart the app. The new visualisation should now be included.