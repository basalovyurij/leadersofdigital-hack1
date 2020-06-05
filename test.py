import csv
import nltk
import roman
from sklearn.feature_extraction.text import CountVectorizer


BAD_WORDS = {'мцк', 'карте', 'посмотреть', 'вднх'}

with open('metro.csv', 'r') as f:
    METROS = set()
    for line in f.readlines():
        METROS.add(line.strip().lower())


def transform_word(s):
    try:
        return str(roman.fromRoman(s.upper()))
    except roman.InvalidRomanNumeralError:
        pass

    if len(s) == 3 and s.endswith('ао'):
        return None

    if s in METROS:
        return None

    if s in BAD_WORDS:
        return None

    return s


def transform_address(address):
    words = nltk.word_tokenize(address, language='russian', preserve_line=True)
    res = ' '.join(filter(lambda x: x is not None, map(transform_word, words)))
    return res


def get_top_words(name, transform):
    res = []
    with open(name, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            address = row[1].lower()
            if transform:
                address = transform_address(address)
            res.append(address)
    vec = CountVectorizer()
    bag_of_words = vec.fit_transform(res)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = dict([(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()])
    return words_freq


bad = get_top_words('bad.csv', True)
good = get_top_words('good.csv', False)

delta = []
for k, v in bad.items():
    if v < 1000:
        continue
    freq = 1
    if k in good:
        freq = good[k]
    delta.append((k, v / freq, v, freq))

delta.sort(key=lambda x: -x[1])
for i in delta[:30]:
    print(i)