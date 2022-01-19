import os
import torch

MODEL_DIR = os.path.join(os.curdir, 'models')
DEFAULT_MODELS = ['sa-distilbert', 'tm-lda', 'am-distilbert']
AVAILABLE_MODELS = ['sa-distilbert', 'tm-lda', 'am-distilbert']
MODEL_CLASS_DICT = {'sa-distilbert': 'DistilBertSentimentAnalysis',
                    'tm-lda' :'LDATopicModeling',
                    'am-distilbert': 'DistilBertArgumentMining'}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
