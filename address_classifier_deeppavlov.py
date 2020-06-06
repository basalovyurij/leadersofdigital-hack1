import html
import re
import pandas
import numpy as np

import pandas as pd
from deeppavlov import *

import test

ROOT_PATH = "C:/Repos/leadersofdigital-hack1"


def generate_train_data():
    good_heads = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
    bad = pandas.read_csv("bad.csv", delimiter=";")['address'].sample(frac=1)
    good = pandas.read_csv("good.csv", delimiter=";", names=good_heads)["2"].sample(frac=1)

    train_good = good[:150000]
    train_bad = bad[:130000]

    train_data = []
    for i in train_good:
        train_data.append([i, 1])
    for i in train_bad:
        train_data.append([i, 0])

    np.random.shuffle(train_data)
    df = pandas.DataFrame(data=train_data, columns=["text", "label"])
    df.to_csv('train_data.csv', sep='\t', encoding='utf-8', index=False)


def normalize_phrase(s):
    # remove special characters
    r = html.unescape(s).lower().replace('\t', ' ').replace('\n', ' ').replace('&', '')
    # remove end numbers
    r = re.sub(r'\s{5,}\d+(\.\d+)?', '', r)
    # remove whitespaces
    r = re.sub(r'\s+', ' ', r)
    return r


settings = {
    "dataset_reader": {
        "class_name": "basic_classification_reader",
        "x": "text",
        "y": "label",
        "data_path": ".",
        "train": "train_data.csv",
        "sep": '\t',
        "class_sep": ','
    },
    "dataset_iterator": {
        "class_name": "basic_classification_iterator",
        "seed": 42,
        "field_to_split": "train",
        "split_fields": [
            "train",
            "valid",
            "test"
        ],
        "split_proportions": [
            0.8,
            0.1,
            0.1
        ]
    },
    "chainer": {
        "in": [
            "x"
        ],
        "in_y": [
            "y"
        ],
        "pipe": [
            {
                "id": "classes_vocab",
                "class_name": "simple_vocab",
                "fit_on": [
                    "y"
                ],
                "save_path": "{MODEL_PATH}/classes.dict",
                "load_path": "{MODEL_PATH}/classes.dict",
                "in": "y",
                "out": "y_ids",
                "special_tokens": ["<UNK>"]
            },
            {
                "in": [
                    "x"
                ],
                "out": [
                    "x_prep"
                ],
                "class_name": "dirty_comments_preprocessor"
            },
            {
                "in": "x_prep",
                "out": "x_tok",
                "id": "my_tokenizer",
                "class_name": "nltk_tokenizer",
                "tokenizer": "wordpunct_tokenize"
            },
            {
                "in": "x_tok",
                "out": "x_emb",
                "id": "my_embedder",
                "class_name": "fasttext",
                "load_path": "{DOWNLOADS_PATH}/embeddings/lenta_lower_100.bin",
                "pad_zero": True
            },
            {
                "in": "y_ids",
                "out": "y_onehot",
                "class_name": "one_hotter",
                "id": "my_one_hotter",
                "depth": "#classes_vocab.len",
                "single_vector": True
            },
            {
                "in": [
                    "x_emb"
                ],
                "in_y": [
                    "y_onehot"
                ],
                "out": [
                    "y_pred_probas"
                ],
                "main": True,
                "class_name": "keras_classification_model",
                "save_path": "{MODEL_PATH}/model",
                "load_path": "{MODEL_PATH}/model",
                "embedding_size": "#my_embedder.dim",
                "n_classes": "#classes_vocab.len",
                "kernel_sizes_cnn": [
                    3,
                    5,
                    7
                ],
                "filters_cnn": 256,
                "optimizer": "Adam",
                "learning_rate": [0.01, 1e-4],
                "learning_rate_decay": "exponential",
                "learning_rate_decay_batches": 5000,
                "learning_rate_drop_patience": 5,
                "learning_rate_drop_div": 5.0,
                "loss": "binary_crossentropy",
                "last_layer_activation": "softmax",
                "coef_reg_cnn": 1e-3,
                "coef_reg_den": 1e-2,
                "dropout_rate": 0.5,
                "dense_size": 100,
                "model_name": "cnn_model"
            },
            {
                "in": "y_pred_probas",
                "out": "y_pred_ids",
                "class_name": "proba2labels",
                "max_proba": True
            },
            {
                "in": "y_pred_ids",
                "out": "y_pred_labels",
                "ref": "classes_vocab"
            }
        ],
        "out": [
            "y_pred_labels"
        ]
    },
    "train": {
        "epochs": 20,
        "batch_size": 64,
        "metrics": [
            {
                "name": "sets_accuracy",
                "inputs": [
                    "y",
                    "y_pred_labels"
                ]
            },
            {
                "name": "roc_auc",
                "inputs": [
                    "y_onehot",
                    "y_pred_probas"
                ]
            }
        ],
        "validation_patience": 5,
        "val_every_n_epochs": 1,
        "log_every_n_epochs": 1,
        "show_examples": False,
        "evaluation_targets": [
            "train",
            "valid",
            "test"
        ],
        "class_name": "nn_trainer"
    },
    "metadata": {
        "variables": {
            "ROOT_PATH": ROOT_PATH,
            "DOWNLOADS_PATH": "{ROOT_PATH}/downloads",
            "MODELS_PATH": "{ROOT_PATH}/models",
            "MODEL_PATH": "{MODELS_PATH}/classifiers/address"
        },
        "requirements": [],
        "download": [
            {
                "url": "http://files.deeppavlov.ai/embeddings/lenta_lower_100.bin",
                "subdir": "{DOWNLOADS_PATH}/embeddings"
            }
        ]
    }
}

# generate_train_data()
# model = train_model(settings, download=True)
model = build_model(settings, download=True)
bad = pd.read_csv("bad.csv", delimiter=";")['address'].values
bad_transformed = [test.transform_address(i) for i in bad]
print(sum(map(lambda x: int(x[0]), model(bad))) / len(bad))
print(sum(map(lambda x: int(x[0]), model(bad_transformed))) / len(bad))
