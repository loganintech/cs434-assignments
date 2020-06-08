
with open('./data/train.csv', "r", encoding="utf-8") as f:

    lines = f.readlines()
    without_header = list(map(lambda line: line.split(",")[0], lines[1:]))
    training_data = without_header[:30000]
    validation_data = without_header[30000:40000]
    test_data = without_header[40000:]
    assert len(training_data) == 30000
    assert len(validation_data) == 10000
    assert len(test_data) == 10000

with open('data/IMDB_labels.csv', "r", encoding='utf-8') as f:
    lines = f.readlines()
    lines = list(map(lambda line: line.split(",")
                     [0] == "positive\n", lines[1:]))
    training_labels = lines[:30000]
    validation_labels = lines[30000:]
    assert len(training_labels) == 30000
    assert len(validation_labels) == 10000
    assert len(lines[40000:]) == 0




def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]a
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    # pattern = r'[^a-zA-z0-9\s]'
    # text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text