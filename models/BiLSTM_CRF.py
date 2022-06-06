import torch
import torch.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM_CRF_NER(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, embed_dim=300, hidden_dim=300, num_layers=3):
        super(BiLSTM_CRF_NER, self).__init__()
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(len(sent_vocab), embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim * 2, len(tag_vocab))

        self.transitions = nn.Parameter(torch.rand(len(tag_vocab), len(tag_vocab)))
        self.transitions.data[tag_vocab.stoi[BOS], :] = -10000
        self.transitions.data[:, tag_vocab.stoi[EOS]] = -10000

    def forward(self, sentences, tags, sent_lengths):
        mask = (sentences != self.sent_vocab.stoi[PAD]).to(self.device)
        sentences = sentences.transpose(0, 1)
        sentences = self.embedding(sentences)
        emit_score = self._get_lstm_features(sentences, sent_lengths)
        loss = self._score_sentence(emit_score, tags, mask)
        return loss

    def _get_lstm_features(self, sentences, sent_lengths):
        padded_sentences = pack_padded_sequence(sentences, sent_lengths, enforce_sorted=False)
        lstm_out, _ = self.lstm(padded_sentences)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        emit_score = self.linear(lstm_out)
        return emit_score

    def _score_sentence(self, emit_score, tags, mask):
        batch_size, sent_len = tags.size()

        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
        score[:, 1:] += self.transitions[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)

        d = torch.unsqueeze(emit_score[:, 0], dim=1)
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[:n_unfinished]
            emit_and_transition = emit_score[:n_unfinished, i].unsqueeze(dim=1) + self.transitions
            log_sum = d_uf.transpose(1, 2) + emit_and_transition
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
            log_sum = log_sum - max_v
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)
        max_d = d.max(dim=-1)[0]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
        llk = total_score - d
        loss = -llk
        return loss

    def predict(self, sentences, sent_lengths):
        batch_size = sentences.shape[0]
        mask = (sentences != self.sent_vocab.stoi[PAD])
        sentences = sentences.transpose(0, 1)
        sentences = self.embedding(sentences)
        emit_score = self._get_lstm_features(sentences, sent_lengths)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size
        d = torch.unsqueeze(emit_score[:, 0], dim=1)
        max_len = max(sent_lengths)
        for i in range(1, max_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]
            emit_and_transition = self.transitions + emit_score[: n_unfinished, i].unsqueeze(dim=1)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)
        _, max_idx = torch.max(d, dim=1)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    def iob_tag(self, tags):
        tags = [self.tag_vocab.itos[tag] for tag in tags]
        prev_tag = 'O'
        for idx, curr_tag in enumerate(tags):
            if curr_tag != 'O' and prev_tag == 'O':
                tags[idx] = 'B-' + curr_tag
            if curr_tag == prev_tag and prev_tag != 'O':
                tags[idx] = 'I-' + curr_tag
            prev_tag = curr_tag
        return tags

    def save(self, filepath):
        params = {}
        params['sent_vocab'] = self.sent_vocab
        params['tag_vocab'] = self.tag_vocab
        params['embed_dim'] = self.embed_dim
        params['hidden_dim'] = self.hidden_dim

        params['embedding'] = self.embedding.state_dict()
        params['lstm'] = self.lstm.state_dict()
        params['linear'] = self.linear.state_dict()
        params['transitions'] = self.transitions
        torch.save(params, filepath)

    @classmethod
    def load(cls, filepath):
        params = torch.load(filepath, map_location=torch.device('cpu'))
        sent_vocab = params['sent_vocab']
        tag_vocab = params['tag_vocab']
        embed_dim = params['embed_dim']
        hidden_dim = params['hidden_dim']

        model = cls(sent_vocab, tag_vocab, embed_dim, hidden_dim)
        model.embedding.load_state_dict(params['embedding'])
        model.lstm.load_state_dict(params['lstm'])
        model.linear.load_state_dict(params['linear'])
        model.transitions = params['transitions']
        return model

    @property
    def device(self):
        return self.embedding.weight.device
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
def padding(sents, pad_idx, device):
    lengths = [len(sent) for sent in sents]
    max_len = lengths[0]
    padded_data = []
    for s in sents:
        padded_data.append(s.tolist() + [pad_idx] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


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


def collate_fn(samples):
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    sentences = [x[0] for x in samples]
    tags = [x[1] for x in samples]
    return sentences, tags
params = torch.load('/home/vanhocvp/Code/AI/NLP/NER/demo/models/weights/vocab_label.pt')
vocab = params['vocab']
label = params['label']
device = torch.device("cpu")

PAD = '<pad>'
UNK = '<unk>'
BOS = '<bos>'
EOS = '<eos>'
def prepare(sents):
    X = []
    for sent in sents:
            x = [vocab.stoi[BOS]]
            for w in sent.split():
                x.append(vocab.stoi[w])
            x.append(vocab.stoi[BOS])
            X.append(torch.tensor(x))
    return X
def evaluate_BiLSTM_CRF(model, sent):
    sentences, sent_lengths = padding(prepare([sent]), model.tag_vocab.stoi[PAD], device)
    pred_tags = model.predict(sentences, sent_lengths)
    # pred_tags = torch.argmax(pred_tags.squeeze(0), dim=1)
    pred_tags = torch.Tensor(pred_tags[0]).int()
    print (pred_tags)
    pred_tags = model.iob_tag(pred_tags.tolist()[1:-1])
    res = ""
    for ind, w in enumerate(sent.split()):
        if pred_tags[ind] != 'O':
            res += w+"[" + pred_tags[ind] + "] "
        else:
            res += w + " "
    return res
def load_BiLSTM_CRF(PATH_WEIGHT):
    return BiLSTM_CRF_NER.load(PATH_WEIGHT)