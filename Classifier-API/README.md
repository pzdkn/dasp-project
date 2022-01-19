# Classifcation API
## Description
* Different Models for Sentiment Analysis, Topic Modeling or Argument Mining can be loaded
* When the API is started a predefined list of default models is loaded for each of the NLP tasks
* A model is specified by its *model-id*
    * e.g. `sa-distilbert` for sentiment analsis
    * e.g `tm-lda` for topic modelling
    * e.g. `am-distilb ert` for argument mining
## Adding a new model
* Add a new model by implementing a new model class in *models.py*
* This class has to inherit from abstract *classifier* class and implement functions for prediction and model loading.
* It has to also inherit from one of the following classes : `SentimentAnalysis`, `ArgumentMining`, `TopicModeling`.
    This is done to guarantee the correct output format for the corresponding prediction types.
* Different convenience functions are already implemented for gpu-prediction and model loading for pytorch-models in *utils.py*
* Once the model class has being implemented, update the list of availabel models in `AVAILABLE_MODELS` and the models dictionary in `MODEL_CLASS_DICT` in file *configs.py*
## Routes
* `predict` make predictions. It takes as input a *json* with fields `model-id` (as specified above) and `text` (a list of strings)
* `get_meta_info` returns a json with meta info for that model (hyperparameters, performance evaluation etc)
* `get_available_models` returns a list of available models
## Basic Usage
* to use our finetuned models, please use the download link provided in *api/models_download_link.txt* and extract it into the *api* folder such that
it respects the folder structure specified below. The models are supplied via download link to avoid adding large files to git history.
* alternatively, models can be downloaded directly from slurm with f.e. : `scp -r pzhao@slurm.ukp.informatik.tu-darmstadt.de:/ukp-storage-1/pzhao/dasp-classification-api/api/models/am/distilbert /<your_local_path>/Classifier-API/api/models/am/distilbert`.
* activate the virtual environment
* run *run.sh* from *api/*
## Folder Structure
Whenever a new model is trained, all its corresponding information is saved according to this folder structure :
```
classification-api
│   README.md
│   requirements.txt    
│
└───api
│   │   ...
│   └───models
│       │
│       └─── am
│       │     │ meta_info.json
│       │     │ pytorch_model.bin
│       │     │ eval_results
│       │     │ ...                      
│       │     └───distilbert
│       │
│       └─── sa
│       │     │
│       │     └─── roberta
│       │
│       └─── tm
│             │
│             └─── lda
│   
└───training
    │   file021.txt
    │   file022.txt
```
For example in `./api/model/am/distilbert` everything produced during training is saved here. Other functions utilize this folder structure, it is important to keep the naming convention <model-type>/<model-name> consistent.
