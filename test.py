import csv

from address_classifier import train_sklearn, SklearnChecker, test_transformer
from simple_transformer import transform_address_simple


def save_result_to_csv(name, result_name):
    res = []
    with open(name, 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
        firstrow = True
        for row in reader:
            if firstrow:
                firstrow = False
            else:
                address = transform_address_simple(row[1])
                res.append([row[0], row[1], address])
    with open(result_name, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(res)


# train_sklearn()
# checker = SklearnChecker('models/lr1.model', 'models/vectorizer1.model')
# test_transformer(transform_address_simple, checker)
save_result_to_csv('bad.csv', 'result_firmachi.csv')
