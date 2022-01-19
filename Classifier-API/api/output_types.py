"""
In output_types.py, the expected output format is implemented for each prediction-type. They are implemented as static
decorators, that implemeted models in models.py can use for the predict function.
"""
class SentimentAnalysis:
    def __init__(self, model_id):
        self.model_id = model_id
        self.label_list = ["negative", "positive", "neutral"]
        self.label_map = {i: label for i, label in enumerate(self.label_list)}

    @staticmethod
    def predict_dec(predict_func):
        def predict(self, data):
            # sentiments are a list of integers corresponding to sentiment types
            # probabilities are a list of dicts containing probabilities for each label
            sentiments, probabilities = predict_func(self, data)
            sentiments = [self.label_list[sent] for sent in sentiments]
            return {"sentiments": sentiments, "probabilities": probabilities}
        return predict


class ArgumentMining:
    def __init__(self, model_id):
        self.model_id = model_id
        self.label_list = ["NoArgument", "Argument_against", "Argument_for"]
        self.label_map = {i: label for i, label in enumerate(self.label_list)}

    @staticmethod
    def predict_dec(predict_func):
        def predict(self, data):
            argument_types, probabilities = predict_func(self, data)
            argument_types = [self.label_list[argument_type] for argument_type in argument_types]
            return {"argument_types": argument_types, "probabilities": probabilities}
        return predict


class TopicModeling:
    def __init__(self, model_id):
        self.model_id = model_id

    @staticmethod
    def predict_dec(predict_func):
        def predict(self, data):
            vocab, doc_lengths, term_frequency, doc_topic_dists, topic_term_dists = predict_func(self, data)
            return {"vocab": vocab, "doc_lengths": doc_lengths,
                    "term_frequency": term_frequency, "topic_term_dists": topic_term_dists,
                    "doc_topic_dists": doc_topic_dists}
        return predict
