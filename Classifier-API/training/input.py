import logging
import os
import glob
from abc import ABC
from typing import List, Optional, Union
from transformers.data.processors import DataProcessor, InputExample, InputFeatures
from transformers.file_utils import is_tf_available
from transformers.tokenization_utils import PreTrainedTokenizer
import csv

logger = logging.getLogger(__name__)

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def convert_examples_to_features(
        examples: Union[List[InputExample], "tf.data.Dataset"],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling convert_examples_to_features from TF, the task parameter is required.")
        return _tf_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    return _convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list)


if is_tf_available():

    def _tf_convert_examples_to_features(
            examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, task=str, max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )


def _convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        else:
            return label_map[example.label]

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class ArgumentMiningProcessor(DataProcessor):

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_train_examples(self, data_dir):
        examples = []
        for tsv_file in glob.glob(os.path.join(data_dir) + '/*.tsv'):
            lines = self._read_tsv(tsv_file)
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                split = line[-1]
                if split == 'test':
                    continue
                guid = "%s-%s" % ("train", i)
                text_a = line[4]  # text
                text_b = line[0]  # topic
                label = line[-2]
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        examples = []
        for tsv_file in glob.glob(os.path.join(data_dir) + '/*.tsv'):
            lines = self._read_tsv(tsv_file)
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                split = line[-1]
                if split == 'train' or split == 'val':
                    continue
                guid = "%s-%s" % ("test", i)
                text_a = line[4]  # text
                text_b = line[0]  # topic
                label = line[-2]
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        return ["NoArgument", "Argument_against", "Argument_for"]


class Sentiment140(DataProcessor):
    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_train_examples(self, data_dir):
        examples = []
        lines = self._read_csv(os.path.join(data_dir, 'training.1600000.processed.noemoticon.csv'))
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            polarity = int(line[0][1])
            guid = "%s-%s" % ("train", i)
            text_a = line[-1]  # text
            if polarity < 2:
                label = 'negative'
            elif polarity == 2:
                label = 'neutral'
            elif polarity > 2:
                label = 'positive'
            assert isinstance(text_a, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_test_examples(self, data_dir):
        examples = []
        lines = self._read_csv(os.path.join(data_dir, 'testdata.manual.2009.06.14.csv'))
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            polarity = int(line[0][1])
            guid = "%s-%s" % ("train", i)
            text_a = line[-1]  # text
            if polarity < 2:
                label = 'negative'
            elif polarity == 2:
                label = 'neutral'
            elif polarity > 2:
                label = 'positive'
            assert isinstance(text_a, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def _read_csv(self, input_file, quotechar=None):
        with open(input_file, "r", encoding="latin-1") as f:
            return list(csv.reader(f, quotechar=quotechar))

    def get_labels(self):
        """See base class."""
        return ["positive", "neutral", "negative"]


class XnliProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_train_examples(self, data_dir):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-MT-1.0/multinli/multinli.train.{}.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2] == "contradictory" else line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != self.language:
                continue
            guid = "%s-%s" % ("test", i)
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


processors = {
    "argument_mining": ArgumentMiningProcessor,
    "sentiment_analysis": Sentiment140
}

if __name__ == '__main__':
    # test if labels have being correctly read
    arg_processor = ArgumentMiningProcessor(language='en')
    examples = arg_processor.get_train_examples(os.path.join('data', 'ArgumentMining'))
    assert all([example.label in arg_processor.get_labels() for example in examples])
    from transformers import DistilBertTokenizer

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    features = convert_examples_to_features(
        examples, tokenizer, max_length=512, label_list=arg_processor.get_labels(),
    )
    sent140 = Sentiment140(language='en')
    examples = sent140.get_train_examples(os.path.join('data', 'SentimentAnalysis'))
    pass
