from configs import MODEL_CLASS_DICT
import models as models


class ClassificationAbstraction:
    """
    This class abstracts from implementation details of the models for the api. At instantiation time, a list of models
    are loaded. To avoid blowing up with too many models,  old models are replaced by new models when they have the
    same type e.g tm, sa or am
    """
    def __init__(self, model_ids):
        self.model_ids = model_ids
        self.models = {model_id:  create_classifier(model_id) for model_id in model_ids}
        pass

    def get_response(self, model_id, data):
        """
        :param model_id: string
        :param data: list of strings
        :return:
        """
        return self.models[model_id].predict(data)

    def add_model(self, model_id):
        """
        replaces existing model of the same type with the new model specified by model_id
        :param model_id:
        :return:
        """
        new_model_type = model_id.split.split('-')
        for existing_model_id in self.model_ids:
            existing_model_type = existing_model_id.split('-')
            if new_model_type == existing_model_type:
                del self.models[existing_model_id]
                self.models.update({model_id: create_classifier(model_id)})
                self.model_ids = [model_id for model_id in self.models.keys()]

    def get_current_model_ids(self):
        return [model_id for model_id in self.models.keys()]


def create_classifier(model_id):
    """
    loads the model constructor from models.py by specifying the model-id and instantiates the corresponding model object
    :param model_id:
    :return: the correpsonding model object
    """
    model_class = getattr(models, MODEL_CLASS_DICT[model_id])
    return model_class(model_id)


if __name__ == '__main__':
    classification_abstraction = ClassificationAbstraction(['sa-distilbert'])
    response = classification_abstraction.get_response('sa-distilbert', ['good', 'bad'])
    print(response)


