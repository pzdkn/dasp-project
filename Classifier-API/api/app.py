from flask import Flask, request, jsonify, make_response
from classification_abstraction import ClassificationAbstraction
from utils import create_meta_info
import configs as configs
import os
import json

# preloads a default set of models at start time, defined in default models
app = Flask(__name__)
classification_abstraction = ClassificationAbstraction(configs.DEFAULT_MODELS)


@app.route('/predict', methods=['POST'])
def predict():
    """
    prediction route, receives two parameters from json
    * model-id: specifies which model should be used. model-id follows syntax : <prediction-type>-<model-name>
                f.e 'sa-distilbert' for sentiment-anaylsis using distilbert
    * text: a list of strings specifying the sentences of the uploaded data
    :return: prediction format varies depending on the prediction type. Output-format is specified in output_types.py
    """
    if request.method == 'POST':
        data = request.get_json()
        model_id = data['model-id']
        if model_id not in configs.AVAILABLE_MODELS:
            return make_response("Specified model can't be found", "404")
        else:
            if model_id not in classification_abstraction.get_current_model_ids():
                classification_abstraction.add_model(model_id)
            text = data['text']
            response = classification_abstraction.get_response(model_id, text)
    return jsonify(response)


@app.route('/meta_info', methods=['POST'])
def get_meta_info():
    """
    returns meta-info for the specified model. Meta-Info is fetched from the model-folders specified in configs.py
    The meta-info is stored in meta_info.json in the corresponding model-folder. If the json-file is not available,
    a new meta_info.json is compiled.
    :return:
    """
    data = request.get_json()
    model_id = data['model-id']
    if model_id not in configs.AVAILABLE_MODELS:
        return make_response("Specified model can't be found", "404")
    else:
        model_type = model_id.split('-')[0]
        model_name = model_id.split('-')[-1]
        model_path = os.path.join(configs.MODEL_DIR, model_type, model_name)
        if not os.path.exists(os.path.join(model_path, 'meta_info.json')):
            create_meta_info(model_path)
        with open(os.path.join(model_path, 'meta_info.json')) as meta_info_json:
            return json.load(meta_info_json)


@app.route('/available_models', methods=['GET'])
def get_available_models():
    """
    returns a list of available models
    :return:
    """
    if request.method == 'GET':
        return jsonify({'available-models': configs.AVAILABLE_MODELS})


def custom_error(message, status_code):
    return make_response(jsonify(message), status_code)


if __name__ == '__main__':
    app.run(debug=True, port=5005)
