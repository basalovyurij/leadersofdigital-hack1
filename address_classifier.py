from joblib import dump, load
import pandas
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, accuracy_score


class Checker:
    def is_good(self, addreses):
        raise NotImplementedError()


def train_sklearn():
    good_heads = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]
    bad = pandas.read_csv("bad.csv", delimiter=";")['address'].sample(frac=1)
    good = pandas.read_csv("good.csv", delimiter=";", names=good_heads)["2"].sample(frac=1)

    train_good = good[:130000]
    train_bad = bad[:130000]
    test_good = good[130000:]
    test_bad = bad[130000:]

    train_data = []
    for i in train_good:
        train_data.append([i, 1])
    for i in train_bad:
        train_data.append([i, 0])

    test_data = []
    for i in test_good:
        test_data.append([i, 1])
    for i in test_bad:
        test_data.append([i, 0])

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    train_x = []
    train_y = []
    for i in train_data:
        train_x.append(i[0])
        train_y.append(i[1])

    test_x = []
    test_y = []
    for i in test_data:
        test_x.append(i[0])
        test_y.append(i[1])

    vectorizer = TfidfVectorizer(min_df=5)
    train_x = vectorizer.fit_transform(train_x)
    test_x = vectorizer.transform(test_x)

    model = LogisticRegression(random_state=42)
    # model = GradientBoostingClassifier(n_estimators=250, random_state=42, verbose=1, max_features='sqrt')
    # model = RandomForestClassifier(n_estimators=10, verbose=1, random_state=241, n_jobs=-1, max_features='sqrt')
    model.fit(train_x, train_y)

    # scores_train = list(map(lambda i: roc_auc_score(train_y, i[:, 1]), list(model.staged_predict_proba(train_x))))
    # scores_test = list(map(lambda i: roc_auc_score(test_y, i[:, 1]), list(model.staged_predict_proba(test_x))))

    # scores_train = list(model.staged_predict_proba(train_x))
    # scores_test = list(model.staged_predict_proba(test_x))

    # plt.figure()
    # plt.plot(scores_train, 'r', linewidth=2)
    # plt.plot(scores_test, 'g', linewidth=2)
    # plt.legend(['test', 'train'])
    # plt.show()

    # score = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    pred = model.predict(test_x)
    score_f1 = f1_score(test_y, pred)
    score_recall = recall_score(test_y, pred)
    score_accuracy = accuracy_score(test_y, pred)

    print("f1:       ", score_f1)
    print("recall:   ", score_recall)
    print("accuracy: ", score_accuracy)

    if not os.path.exists('models'):
        os.makedirs('models')
    dump(model, 'models/lr1.model')
    dump(vectorizer, 'models/vectorizer1.model')


class SklearnChecker(Checker):
    def __init__(self, model_path, vectorizer_path):
        self.model = load(model_path)
        self.vectorizer = load(vectorizer_path)

    def is_good(self, addreses):
        return self.model.predict(self.vectorizer.transform(addreses))


def test_transformer(transform_func, checker: Checker):
    bad_values = pandas.read_csv("bad.csv", delimiter=";")['address'].tolist()
    pred_before_transform = checker.is_good(bad_values)
    print(sum(pred_before_transform) / len(bad_values))

    transformed = [transform_func(i) for i in bad_values]
    pred_after_transform = checker.is_good(transformed)
    print(sum(pred_after_transform) / len(bad_values))
