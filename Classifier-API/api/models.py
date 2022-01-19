from abc import ABC, abstractmethod
from output_types import SentimentAnalysis, ArgumentMining, TopicModeling
from utils import pytorch_load_model_dec, pytorch_predict_gpu, convert_sentences_to_dataset, load_transformers_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import configs
import numpy as np
import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(ABC):
    """
    Abstract classifier class, which every models has to inherit. It specifies the required functions, predict and
    load_model. **Note that this is independent of which library is used for the classifier, f.e. tf or pytorch.
    As long as both methods are implemented, this should work fine.
    """
    @abstractmethod
    def predict(self, data):
        """
        abstract method that every child class needs to implement
        :param data: list of strings
        :return:
        """
        pass

    @abstractmethod
    def load_model(self, model_id):
        """
        loads model according to model_type and model_id.
        :param model_id: string
        :return: model object
        """
        pass


class DistilBertSentimentAnalysis(Classifier, SentimentAnalysis):
    def __init__(self, model_id):
        SentimentAnalysis.__init__(self, model_id)
        self.model, self.tokenizer = self.load_model()

    @pytorch_load_model_dec
    def load_model(self):
        from transformers import (DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)

        config = DistilBertConfig.from_pretrained('distilbert-base-uncased',
                                                  num_labels=len(self.label_list))
        model = DistilBertForSequenceClassification(config)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        return model, tokenizer

    @SentimentAnalysis.predict_dec
    def predict(self, data):
        preds, probabilities = pytorch_predict_gpu(self, data)
        return preds, probabilities


class DistilBertArgumentMining(Classifier, ArgumentMining):
    def __init__(self, model_id):
        ArgumentMining.__init__(self, model_id)
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        model, tokenizer = load_transformers_model(self.model_id)
        return model, tokenizer
    @ArgumentMining.predict_dec
    def predict(self, data):
        return pytorch_predict_gpu(self, data)


class LDATopicModeling(Classifier, TopicModeling):
    def __init__(self, model_id):
        TopicModeling.__init__(self, model_id)
        self.vectorizer, self.model = self.load_model(model_id)
        self.count_data = None

    def load_model(self, model_id):
        vectorizer = CountVectorizer()
        model = LatentDirichletAllocation(n_components=10, n_jobs=-1,
                                          learning_method='online')
        return vectorizer, model

    @TopicModeling.predict_dec
    def predict(self, data):
        "For more info see [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html]"
        self.count_data = self.vectorizer.fit_transform(data)
        self.model.fit(self.count_data)
        vocab = self.vectorizer.get_feature_names()
        doc_lengths = self.count_data.sum(axis=1).getA1()
        term_frequency = self.count_data.sum(axis=0).getA1()
        transformed = self.model.transform(self.count_data)
        doc_topic_dists = transformed / transformed.sum(axis=1)[:, np.newaxis]
        # components_[i,j] describe the number of times word j was ascribed to topic i
        topic_term_dists = self.model.components_ / self.model.components_.sum(axis=1)[:, np.newaxis]
        self.evaluate()
        return vocab, doc_lengths.tolist(), term_frequency.tolist(), doc_topic_dists.tolist(), topic_term_dists.tolist()

    def evaluate(self):
        model_type = self.model_id.split('-')[0]
        model_name = self.model_id.split('-')[-1]
        eval_path = os.path.join(configs.MODEL_DIR, model_type, model_name, 'eval_results.txt')
        loglikelihood = self.model.score(self.count_data)
        perplexity = self.model.perplexity(self.count_data)
        with open(eval_path, 'w') as writer:
            writer.write(" %s = %s \n" % ('loglikelihood', loglikelihood))
            writer.write(" %s = %s \n" % ('perplexity', perplexity))


if __name__ == '__main__':
    # test lda model
    tm = LDATopicModeling('tm-lda')
    data = ['This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?']
    tm_output = tm.predict(data)
    sentences = ["Hello World", "Goodbye World"]
    am = DistilBertArgumentMining('am-distilbert')
    sm_output = am.predict(sentences)
    print(sm_output)

