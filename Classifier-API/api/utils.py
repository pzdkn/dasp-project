import torch
import os
import json
import configs
import numpy as np
from transformers.data.processors import DataProcessor, InputExample, InputFeatures
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm

def pytorch_load_model_dec(load_func):
    """
    used to load pytorch models that are only given by .pt files
    :param load_func:
    :return:
    """
    def load_model(self):
        model_type = self.model_id.split('-')[0]
        model_name = self.model_id.split('-')[-1]
        path = os.path.join(configs.MODEL_DIR, model_type, model_name, model_name + '.pt')
        model, tokenizer = load_func(self)
        model.load_state_dict(torch.load(path, map_location=configs.DEVICE))
        model.to(configs.DEVICE)
        model.eval()
        print("Model device: {}".format(next(model.parameters()).device))
        return model, tokenizer

    return load_model


def pytorch_predict_gpu(classifier, data, batch_size=16):
    """
    used for prediction for gpu with pytorch
    :param classifier:
    :param data:
    :param batch_size:
    :return:
    """
    model_type = classifier.model_id.split('-')[-1]
    dataset = convert_sentences_to_dataset(classifier, data)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    preds = None
    for batch in tqdm(dataloader, desc='Prediction'):
        classifier.model.eval()
        model_device = next(classifier.model.parameters()).device
        batch = tuple(t.to(configs.DEVICE) for t in batch)
        input_device = batch[0].device
        if input_device != model_device:
            raise Exception('Model device is not equal to input device \n'
                            'Model device: {}\n'
                            'Input device: {}'.format(model_device, input_device))
        with torch.no_grad():
            if model_type == "distilbert":
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                inputs["token_type_ids"] = (
                    batch[2] if model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids
            logits = classifier.model(**inputs)[0]
            if preds is None:
                preds = logits.detach().cpu()
            else:
                preds = torch.cat((preds, logits.detach().cpu()), axis=0)
    probabilities = preds.softmax(dim=1).numpy()
    preds = np.argmax(probabilities, axis=1).astype(int)
    preds = preds.tolist()
    probabilities = probabilities.tolist()
    probabilities = [{label:prob for prob, label in zip(probs, classifier.label_list)} for probs in probabilities]
    return preds, probabilities


def convert_sentences_to_dataset(classifier, data, max_length=512):
    """
    converts to input data to a pytorch dataset, used for gpu prediction
    :param classifier:
    :param data:
    :param max_length:
    :return:
    """
    examples = [InputExample(guid=i, text_a=sentence) for i, sentence in enumerate(data)]
    batch_encoding = classifier.tokenizer.batch_encode_plus(
        [example.text_a for example in examples], max_length=max_length, pad_to_max_length=True,
    )
    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs)
        features.append(feature)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    model_type = classifier.model_id.split('-')[-1]
    if model_type != 'distilbert':
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask)
    return dataset


def pytorch_predict_cpu(classifier, data):
    """
    convenience function to run predictions on cpu for pytorch
    :param data:
    :return:
    """
    preds = []
    probabilities = []
    with torch.no_grad():
        for sentence in data:
            input_id = torch.tensor([classifier.tokenizer.encode(sentence)])
            output = classifier.model(input_id)
            logit = output[0]
            probs = logit.softmax(dim=1).numpy()
            probs = probs.tolist()[0]
            pred = int(np.argmax(logit.numpy(), axis=1))
            probability = {label: prob for label, prob in zip(classifier.label_list, probs)}
            preds.append(classifier.label_map[pred])
            probabilities.append(probability)
    return preds, probabilities

def load_transformers_model(model_id):
    """
    loads a model that has being trained originating from the transformers library
    :param model_id:
    :return:
    """
    model_type = model_id.split('-')[0]
    model = model_id.split('-')[-1]
    path = os.path.join(configs.MODEL_DIR, model_type, model)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model.to(configs.DEVICE)
    model.eval()
    return model, tokenizer

def create_meta_info(model_path):
    """
    crawling through the model folder and accumulating the model information in a json
    :param model_path:
    :return:
    """
    meta_info = {'performance': None, 'hyperparameters': None, 'training-parameters': None}
    # format the evaluation performance
    eval_path = os.path.join(model_path, 'eval_results.txt')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as eval_txt:
            meta_info['performance'] = {}
            for line in eval_txt:
                metric = line.split()[0]
                meta_info['performance'][metric] = float(line.split()[-1])
    # onto hyperparameters
    hyperparam_path = os.path.join(model_path, 'config.json')
    if os.path.exists(hyperparam_path):
        with open(hyperparam_path, 'r') as hyperparam_json:
            meta_info['hyperparameters'] = json.load(hyperparam_json)
    # now training parameters
    training_param_path = os.path.join(model_path, 'training_args.bin')
    if os.path.exists(training_param_path):
        training_args = vars(torch.load(training_param_path))
        wanted_keys = ['language', 'max_seq_length', 'do_lower_case', 'per_gpu_train_batch_size'
                                                                      'gradient_accumulation_steps', 'learning_rate',
                       'weight_decay',
                       'adam_epsilon', 'max_grad_norm', 'num_train_epochs', 'max_steps',
                       'warmup_steps', 'seed', 'train_batch_size']
        meta_info['training-parameters'] = {k: v for k, v in training_args.items() if k in wanted_keys}
    with open(os.path.join(model_path, 'meta_info.json'), 'w') as meta_info_json:
        json.dump(meta_info, meta_info_json)


if __name__ == '__main__':
    create_meta_info(os.path.join('models', 'am', 'distilbert'))
    pass
