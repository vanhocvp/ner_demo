import shutil

import sklearn_crfsuite
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP(address="http://127.0.0.1", port=8000) 
from string import punctuation

class FeatureExtractor:
    def extract(self, sentences):
        X = [self.sentence2features(s) for s in sentences]
        y = [self.sentence2lables(s) for s in sentences]
        return X,y

    def sentence2features(self, s):
        return [self.word2features(s, i) for i in range(len(s))]

    def sentence2lables(self, s):
        return [row[-1] for row in s]

    def word2features(self, s, i):
        word = s[i][0]
        features = {
            'bias'       : 1.0,
            '[0]'        : word,
            '[0].lower'  : word.lower(),
            '[0].istitle': word.istitle(),
            '[0].isdigit': word.isdigit(),
            '[0].ispunct': word in punctuation,
        }
        if i > 0:
            word1 = s[i - 1][0]
            tag1 = s[i - 1][1]
            features.update({
                '[-1]'        : word1,
                '[-1].lower'  : word1.lower(),
                '[-1].istitle': word1.istitle(),
                '[-1].isdigit': word1.isdigit(),
                '[-1].ispunct': word1 in punctuation,
                '[-1][1]'     : tag1,
                '[-1,0]'      : "%s %s" % (word1, word),
            })
            if i > 1:
                word2 = s[i - 2][0]
                tag2 = s[i - 2][1]
                features.update({
                    '[-2]'        : word2,
                    '[-2].lower'  : word2.lower(),
                    '[-2].istitle': word2.istitle(),
                    '[-2].isdigit': word2.isdigit(),
                    '[-2].ispunct': word2 in punctuation,
                    '[-2][1]'     : tag2,
                    '[-2,-1]'     : "%s %s" % (word2, word1),
                    '[-2,-1][1]'  : "%s %s" % (tag2, tag1),
                })
                if i > 2:
                    tag3 = s[i - 3][1]
                    features.update({
                        '[-3][1]'    : tag3,
                        '[-3,-2][1]' : "%s %s" % (tag3, tag2),
                    })
        else:
            features['BOS'] = True

        if i < len(s) - 1:
            word1 = s[i + 1][0]
            features.update({
                '[+1]'        : word1,
                '[+1].lower'  : word1.lower(),
                '[+1].istitle': word1.istitle(),
                '[+1].isdigit': word1.isdigit(),
                '[+1].ispunct': word1 in punctuation,
                '[0,+1]'      : "%s %s" % (word, word1)
            })
            if i < len(s) - 2:
                word2 = s[i + 2][0]
                features.update({
                    '[+2]'        : word2,
                    '[+2].lower'  : word2.lower(),
                    '[+2].istitle': word2.istitle(),
                    '[+2].isdigit': word2.isdigit(),
                    '[+2].ispunct': word2 in punctuation,
                    '[+1,+2]'     : "%s %s" % (word1, word2)
                })
        else:
            features['EOS'] = True
        return features

class CRF_NER(sklearn_crfsuite.CRF):
    def save(self, model_filename):
        destination = model_filename
        source = self.modelfile.name
        shutil.copy(src=source, dst=destination)

    @staticmethod
    def load(model_filename):
        model = CRF_NER(model_filename=model_filename)
        return model

def _get_tags(sents):
    tags = []
    for sent_idx, iob_tags in enumerate(sents):
        curr_tag = {'type': None, 'start_idx': None,
                    'end_idx': None, 'sent_idx': None}
        for i, tag in enumerate(iob_tags):
            if tag == 'O' and curr_tag['type']:
                tags.append(tuple(curr_tag.values()))
                curr_tag = {'type': None, 'start_idx': None,
                            'end_idx': None, 'sent_idx': None}
            elif tag.startswith('B'):
                curr_tag['type'] = tag[2:]
                curr_tag['start_idx'] = i
                curr_tag['end_idx'] = i
                curr_tag['sent_idx'] = sent_idx
            elif tag.startswith('I'):
                curr_tag['end_idx'] = i
        if curr_tag['type']:
            tags.append(tuple(curr_tag.values()))
    tags = set(tags)
    return tags


def f_measure(y_true, y_pred):
    tags_true = _get_tags(y_true)
    tags_pred = _get_tags(y_pred)

    ne_ref = len(tags_true)
    ne_true = len(set(tags_true).intersection(tags_pred))
    ne_sys = len(tags_pred)
    if ne_ref == 0 or ne_true == 0 or ne_sys == 0:
        return 0
    p = ne_true / ne_sys
    r = ne_true / ne_ref
    f1 = (2 * p * r) / (p + r)

    return f1


def evaluate_CRF(model, sent ):
    

    test_data = []
    l = rdrsegmenter.tokenize(sent)
    print(l)
    line = ""
    for li in l[0]:
        line+=li+" "
    sents = [line]
    for sent in sents:
        x = []
        
        items = sent.split(' ')
        for item in items:
            
            x.append((item, 'O'))
        test_data.append(x)

    feature_extractor = FeatureExtractor()
    X_test, y_test = feature_extractor.extract(test_data)
    y_pred = model.predict(X_test)
    print(y_pred, sents)
    result = ""
    sents = sents[0].split(" ")
    for i, w in enumerate(sents):
                        if i < len(sents)-1:
                                if y_pred[0][i] != "O":
                                    result+= w+ "["+y_pred[0][i]+"] "
                                else:
                                        result += w + " "       
                                
                        elif i == len(sents)-1:
                                if y_pred[0][i] == y_pred[0][i-1] and y_pred[0][i] != "O" :
                                        result += w + "["+y_pred[0][i]+"] "
                                else:
                                        result += w
    print(result)
    return result

model = CRF_NER(
    c1=1.0,
    c2=1e-3,
    max_iterations=400,
    all_possible_transitions=True,
    verbose=True,
)

def load_CRF(PATH_WEIGHT):
    # model = CRF_NER.load('/home/vanhocvp/Code/AI/NLP/NER/demo/models/weights/model1.crfsuite')
    model = CRF_NER.load(PATH_WEIGHT)
    return model
# sent = "Ban Bí thư xác định, ông Sùng Minh Sính với cương vị là Ủy viên Ban Thường vụ Tỉnh ủy, Trưởng ban Nội chính Tỉnh ủy Hà Giang"

# evaluate(model,sent = sent)