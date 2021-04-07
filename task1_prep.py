import csv
import os
from typing import List

import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

LABEL2ID = {'0': 0, '1': 1}


class InputExample(object):
    def __init__(self, nid, topic_id, tweet_id, tweet_url, tweet_text, claim, check_worthiness):
        self.nid = nid
        self.topic_id = topic_id
        self.tweet_id = tweet_id
        self.tweet_url = tweet_url
        self.tweet_text = tweet_text
        self.claim = claim
        self.label = check_worthiness

    def __repr__(self):
        return str(self.__dict__)


class InputTest(object):
    def __init__(self, nid, topic_id, tweet_id, tweet_url, tweet_text, claim):
        self.nid = nid
        self.topic_id = topic_id
        self.tweet_id = tweet_id
        self.tweet_url = tweet_url
        self.tweet_text = tweet_text
        self.claim = claim

    def __repr__(self):
        return str(self.__dict__)


class InputFeatures(object):
    def __init__(self, topic_id_id, tweet_text_id, claim_id, label_id):
        self.topic_id_id = topic_id_id
        self.tweet_text_id = tweet_text_id
        self.claim_id = claim_id
        self.label_id = label_id

    def __repr__(self):
        return str(self.__dict__)


class InputFeaturesTest(object):
    def __init__(self, topic_id_id, tweet_text_id, claim_id):
        self.topic_id_id = topic_id_id
        self.tweet_text_id = tweet_text_id
        self.claim_id = claim_id

    def __repr__(self):
        return str(self.__dict__)


class DataProcessor(object):
    def _read_csv(cls, input_file) -> List[List[str]]:
        with open(input_file, "r", encoding='utf-8') as reader:
            csv_reader = csv.reader(reader, delimiter="\t")
            data = []
            for line in csv_reader:
                data.append(line)
        return data

    def get_train_examples(self, data_dir='./Training_Data'):
        """See base class."""
        data = self._read_csv(os.path.join(
            data_dir, "training.tsv"))[1:]
        answer = self._read_csv(os.path.join(
            data_dir, "answer.tsv"))[1:]
        return self._create_examples((data, answer), "train")

    def get_dev_examples(self, data_dir='./Dev_Data'):
        """See base class."""
        data = self._read_csv(os.path.join(
            data_dir, "dev.tsv"))[1:]
        answer = self._read_csv(os.path.join(
            data_dir, "answer.tsv"))[1:]
        return self._create_examples((data, answer), "dev")

    def get_test_dataset(self, data_dir='./Test_Data'):
        """See base class."""
        data = self._read_csv(os.path.join(
            data_dir, "test.tsv"))[1:]
        return self._create_test((data), "test")

    def get_test_examples(self, data_dir):
        """See base class."""
        raise NotImplementedError('test not released!')

    def get_labels(self, data_dir):
        """See base class."""
        return [0, 1]

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        iterator = 0
        for data, answer in zip(*dataset):
            topic_id, tweet_id, tweet_url, tweet_text, claim = data
            check_worthiness = answer[0]

            examples.append(InputExample(
                nid=iterator, topic_id=topic_id, tweet_id=tweet_id, tweet_url=tweet_url, tweet_text=tweet_text,
                claim=claim, check_worthiness=check_worthiness))
            iterator = iterator + 1
        return examples

    def _create_test(self, dataset, set_type):
        test = []
        iterator = 0
        for data in dataset:
            topic_id, tweet_id, tweet_url, tweet_text = data

            test.append(InputTest(
                nid=iterator, topic_id=topic_id, tweet_id=tweet_id, tweet_url=tweet_url, tweet_text=tweet_text,
                claim='1'))
            iterator = iterator + 1
        return test


def convert_examples_to_features(examples: List[InputExample], label2id, max_seq_length, tokenizer) -> List[
    InputFeatures]:
    features = []
    pad_token = tokenizer.pad_token_id

    for (i, example) in enumerate(examples):
        topic_id_tok = example.topic_id
        tweet_text_tok = example.tweet_text
        topic_id_id = tokenizer.encode(topic_id_tok, add_special_tokens=True)
        tweet_text_id = tokenizer.encode(tweet_text_tok, add_special_tokens=True)

        assert len(topic_id_id) < max_seq_length and len(tweet_text_id) < max_seq_length

        padding_length1 = max_seq_length - len(topic_id_id)
        padding_length2 = max_seq_length - len(tweet_text_id)

        topic_id_id = topic_id_id + [pad_token] * padding_length1
        tweet_text_id = tweet_text_id + [pad_token] * padding_length2

        claim_id = label2id[example.claim]
        label_id = label2id[example.label]

        features.append(
            InputFeatures(
                topic_id_id=topic_id_id,
                tweet_text_id=tweet_text_id,
                claim_id=claim_id,
                label_id=label_id
            )
        )

    return features


def convert_test_to_features(examples: List[InputTest], label2id, max_seq_length, tokenizer) -> List[
    InputFeaturesTest]:
    features = []
    pad_token = tokenizer.pad_token_id

    for (i, example) in enumerate(examples):
        topic_id_tok = example.topic_id
        tweet_text_tok = example.tweet_text
        topic_id_id = tokenizer.encode(topic_id_tok, add_special_tokens=True)
        tweet_text_id = tokenizer.encode(tweet_text_tok, add_special_tokens=True)

        assert len(topic_id_id) < max_seq_length and len(
            tweet_text_id) < max_seq_length

        padding_length1 = max_seq_length - len(topic_id_id)
        padding_length2 = max_seq_length - len(tweet_text_id)

        topic_id_id = topic_id_id + [pad_token] * padding_length1
        tweet_text_id = tweet_text_id + [pad_token] * padding_length2

        claim_id = label2id[example.claim]

        features.append(
            InputFeaturesTest(
                topic_id_id=topic_id_id,
                tweet_text_id=tweet_text_id,
                claim_id=claim_id
            )
        )

    return features


def convert_features_to_dataset(features: List[InputFeatures]):
    all_topic_id_id = torch.tensor(
        [f.topic_id_id for f in features], dtype=torch.long)
    all_tweet_text_id = torch.tensor(
        [f.tweet_text_id for f in features], dtype=torch.long)
    all_claim_id = torch.tensor(
        [f.claim_id for f in features], dtype=torch.long)
    all_label_id = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_topic_id_id, all_tweet_text_id, all_claim_id, all_label_id)
    return dataset


def convert_features_to_test_dataset(features: List[InputFeaturesTest]):
    all_topic_id_id = torch.tensor(
        [f.topic_id_id for f in features], dtype=torch.long)
    all_tweet_text_id = torch.tensor(
        [f.tweet_text_id for f in features], dtype=torch.long)
    all_claim_id = torch.tensor(
        [f.claim_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_topic_id_id, all_tweet_text_id, all_claim_id)
    return dataset


if __name__ == "__main__":
    # test
    d = DataProcessor()
    train_dir_path = './Training_Data'
    dev_dir_path = 'Dev_Data'
    dev = d.get_dev_examples(dev_dir_path)

    train = d.get_train_examples(train_dir_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print(dev[1])

    features = convert_examples_to_features(
        examples=train, label2id=LABEL2ID, max_seq_length=128, tokenizer=tokenizer)

    dataset = convert_features_to_dataset(features)
