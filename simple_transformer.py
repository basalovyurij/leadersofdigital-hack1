import csv
import nltk
import roman
import string
import re
from sklearn.feature_extraction.text import CountVectorizer


def read_file(name):
    with open(name, 'r', encoding='utf8') as f:
        res = set()
        for line in f.readlines():
            res.add(line.strip().lower())
    return res


BAD_WORDS = read_file('bad_words.csv')
METROS = read_file('metro.csv')
ADDR_REGEX = r'.*?(((кв|к|стр|д|дом)\.?\s\d{1,4})|([а-яА-Я]{4,}\s[\d\-\/]{1,5}))'


def clear_address(addr):
    match = re.search(ADDR_REGEX, addr)
    return match.group(0) if match else addr


def transform_word(s):
    if s == 'ii':
        return '2'

    try:
        return str(roman.fromRoman(s.upper()))
    except roman.InvalidRomanNumeralError:
        pass

    if len(s) == 3 and s.endswith('ао'):
        return None

    if s in BAD_WORDS:
        return None

    return s.strip('.')


def transform_address_simple(address):
    address = clear_address(address.lower())
    for s in METROS:
        address = address.replace('м.' + s, '').replace('м. ' + s, '')
    words = nltk.word_tokenize(address, language='russian', preserve_line=True)
    res = ' '.join(filter(lambda x: x is not None and x not in string.punctuation, map(transform_word, words)))
    return res


def transform_multi_addresses_simple(addresses):
    return [transform_address_simple(a) for a in addresses]


def get_top_words(name, transform):
    res = []
    with open(name, 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            address = row[1].lower()
            if transform:
                address = transform_address_simple(address)
            res.append(address)
    vec = CountVectorizer()
    bag_of_words = vec.fit_transform(res)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = dict([(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()])
    return words_freq


def generate_bad_words():
    bad = get_top_words('bad.csv', True)
    good = get_top_words('good.csv', False)

    delta = []
    for k, v in bad.items():
        if v < 500:
            continue
        freq = 1
        if k in good:
            freq = good[k]
        delta.append((k, v / freq, v, freq))

    with open('bad_words.csv', 'w', encoding='utf8') as f:
        for i in delta:
            if i[1] > 10:
                f.write(i[0] + '\n')
