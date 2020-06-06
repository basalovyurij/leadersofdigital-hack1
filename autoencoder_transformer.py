import collections
from collections import Counter
import csv
from munch import Munch
import os
import nltk
import numpy as np
import random
import string
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Vocab(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []

        with open(path) as f:
            for line in f:
                w = line.split()[0]
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
        self.size = len(self.word2idx)

        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']

    @staticmethod
    def build(sents, path, size):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        words = [w for s in sents for w in s]
        cnt = Counter(words)
        n_unk = len(words)
        for w, c in cnt.most_common(size):
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk

        with open(path, 'w') as f:
            for w in v:
                f.write('{}\t{}\n'.format(w, cnt[w]))


def word_shuffle(vocab, x, k):   # slight shuffle such that |sigma[i]-i| <= k
    base = torch.arange(x.size(0), dtype=torch.float).repeat(x.size(1), 1).t()
    inc = (k+1) * torch.rand(x.size())
    inc[x == vocab.go] = 0     # do not shuffle the start sentence symbol
    inc[x == vocab.pad] = k+1  # do not shuffle end paddings
    _, sigma = (base + inc).sort(dim=0)
    return x[sigma, torch.arange(x.size(1))]


def word_drop(vocab, x, p):     # drop words with probability p
    x_ = []
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = True  # do not drop the start sentence symbol
        sent = [w for j, w in enumerate(words) if keep[j]]
        sent += [vocab.pad] * (len(words)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)


def word_blank(vocab, x, p):     # blank words with probability p
    blank = (torch.rand(x.size(), device=x.device) < p) & \
        (x != vocab.go) & (x != vocab.pad)
    x_ = x.clone()
    x_[blank] = vocab.blank
    return x_


def word_substitute(vocab, x, p):     # substitute words with probability p
    keep = (torch.rand(x.size(), device=x.device) > p) | \
        (x == vocab.go) | (x == vocab.pad)
    x_ = x.clone()
    x_.random_(0, vocab.size)
    x_[keep] = x[keep]
    return x_


def noisy(vocab, x, drop_prob, blank_prob, sub_prob, shuffle_dist):
    if shuffle_dist > 0:
        x = word_shuffle(vocab, x, shuffle_dist)
    if drop_prob > 0:
        x = word_drop(vocab, x, drop_prob)
    if blank_prob > 0:
        x = word_blank(vocab, x, blank_prob)
    if sub_prob > 0:
        x = word_substitute(vocab, x, sub_prob)
    return x


def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]


def set_seed(seed):     # set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)


def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, vocab, args, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.embed = nn.Embedding(vocab.size, args.dim_emb)
        self.proj = nn.Linear(args.dim_h, vocab.size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)


class DAE(TextModel):
    """Denoising Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.drop = nn.Dropout(args.dropout)
        self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0)
        self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
        self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb)
        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (h, _) = self.E(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg == 'greedy' or alg == 'sample'
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            else:
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, input, is_train=False):
        _input = noisy(self.vocab, input, *self.args.noise) if is_train else input
        mu, logvar = self.encode(_input)
        z = reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, is_train=False):
        _, _, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean()}

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()

    def nll_is(self, inputs, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is


class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets, is_train=False):
        mu, logvar, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}


class AAE(DAE):
    """Adversarial Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
            nn.Linear(args.dim_d, 1), nn.Sigmoid())
        self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy(self.D(zn), ones)
        loss_g = F.binary_cross_entropy(self.D(z), ones)
        return loss_d, loss_g

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_adv * losses['adv'] + \
            self.args.lambda_p * losses['|lvar|']

    def autoenc(self, inputs, targets, is_train=False):
        _, logvar, z, logits = self(inputs, is_train)
        loss_d, adv = self.loss_adv(z)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d}

    def step(self, losses):
        super().step(losses)

        self.optD.zero_grad()
        losses['loss_d'].backward()
        self.optD.step()


def get_batch(x, vocab, device):
    go_x, x_eos = [], []
    max_len = max([len(s) for s in x])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch


def get_batches(data, vocab, batch_size, device):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch(data[i: j], vocab, device))
        i = j
    return batches, order


class AutoEncoder(object):
    def __init__(self, args):
        self.args = args
        self.vocab = Vocab(os.path.join(args.checkpoint, 'vocab.txt'))
        set_seed(args.seed)
        cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.model = self.get_model(os.path.join(args.checkpoint, 'model.pt'))

    def get_model(self, path):
        ckpt = torch.load(path)
        train_args = ckpt['args']
        model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[train_args.model](self.vocab, train_args).to(self.device)
        model.load_state_dict(ckpt['model'])
        model.flatten()
        model.eval()
        return model

    def encode(self, sents):
        assert self.args.enc == 'mu' or self.args.enc == 'z'
        batches, order = get_batches(sents, self.vocab, self.args.batch_size, self.device)
        z = []
        for inputs, _ in batches:
            mu, logvar = self.model.encode(inputs)
            if self.args.enc == 'mu':
                zi = mu
            else:
                zi = reparameterize(mu, logvar)
            z.append(zi.detach().cpu().numpy())
        z = np.concatenate(z, axis=0)
        z_ = np.zeros_like(z)
        z_[np.array(order)] = z
        return z_

    def decode(self, z):
        sents = []
        i = 0
        while i < len(z):
            zi = torch.tensor(z[i: i+self.args.batch_size], device=self.device)
            outputs = self.model.generate(zi, self.args.max_len, self.args.dec).t()
            for s in outputs:
                sents.append([self.vocab.idx2word[id] for id in s[1:]])  # skip <go>
            i += self.args.batch_size
        return strip_eos(sents)


def tokenize(s):
    words = nltk.word_tokenize(s.lower(), language='russian', preserve_line=True)
    return [w for w in words if w not in string.punctuation]


def create_train_valid_test_data():
    data = []
    with open('good.csv', 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            data.append(' '.join(tokenize(row[1])))
            if len(data) == 100000:
                break

    random.shuffle(data)
    l = len(data)

    path = 'autoencoder/data/sb-sm'
    with open(path + 'train.csv', 'w', encoding='utf8') as f:
        f.write('\n'.join(data[:int(0.9 * l)]))
    with open(path + 'valid.csv', 'w', encoding='utf8') as f:
        f.write('\n'.join(data[int(0.9 * l):int(0.95 * l)]))
    with open(path + 'test.csv', 'w', encoding='utf8') as f:
        f.write('\n'.join(data[int(0.95 * l):]))

_def_args = {
    'enc': 'mu',
    'dec': 'greedy',
    'batch_size': 256,
    'max_len': 35,
    'seed': 1111,
    'no_cuda': True
}


def base_factory(args):
    def convert(s):
        return ' '.join([i for i in s if i != '<unk>'])

    def transform(sents):
        res = model.decode(model.encode([tokenize(s) for s in sents]))
        return [convert(s) for s in res]

    model = AutoEncoder(Munch(args))
    return transform


def create_daae_transformer_sm():
    args = _def_args.copy()
    args['checkpoint'] = 'autoencoder/checkpoints/sb-sm/daae'
    return base_factory(args)


class AverageMeter(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.cnt = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.cnt += n
        self.sum += val * n
        self.avg = self.sum / self.cnt


def evaluate(model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for inputs, targets in batches:
            losses = model.autoenc(inputs, targets)
            for k, v in losses.items():
                meters[k].update(v.item(), inputs.size(1))
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters


def logging(s, path, print_=True):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')


def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line.split())
    return sents


def train(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging(str(args), log_file)

    # Prepare data
    train_sents = load_sent(args.train)
    logging('# train sents {}, tokens {}'.format(
        len(train_sents), sum(len(s) for s in train_sents)), log_file)
    valid_sents = load_sent(args.valid)
    logging('# valid sents {}, tokens {}'.format(
        len(valid_sents), sum(len(s) for s in valid_sents)), log_file)
    vocab_file = os.path.join(args.save_dir, 'vocab.txt')
    if not os.path.isfile(vocab_file):
        Vocab.build(train_sents, vocab_file, args.vocab_size)
    vocab = Vocab(vocab_file)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[args.model](
        vocab, args).to(device)
    if args.load_model:
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model'])
        model.flatten()
    logging('# model parameters: {}'.format(
        sum(x.data.nelement() for x in model.parameters())), log_file)

    train_batches, _ = get_batches(train_sents, vocab, args.batch_size, device)
    valid_batches, _ = get_batches(valid_sents, vocab, args.batch_size, device)
    best_val_loss = None
    for epoch in range(args.epochs):
        start_time = time.time()
        logging('-' * 80, log_file)
        model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(train_batches)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            inputs, targets = train_batches[idx]
            losses = model.autoenc(inputs, targets, is_train=True)
            losses['loss'] = model.loss(losses)
            model.step(losses)
            for k, v in losses.items():
                meters[k].update(v.item())

            if (i + 1) % args.log_interval == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(indices))
                for k, meter in meters.items():
                    log_output += ' {} {:.2f},'.format(k, meter.avg)
                    meter.clear()
                logging(log_output, log_file)

        valid_meters = evaluate(model, valid_batches)
        logging('-' * 80, log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            log_output += ' {} {:.2f},'.format(k, meter.avg)
        if not best_val_loss or valid_meters['loss'].avg < best_val_loss:
            log_output += ' | saving model'
            ckpt = {'args': args, 'model': model.state_dict()}
            torch.save(ckpt, os.path.join(args.save_dir, 'model.pt'))
            best_val_loss = valid_meters['loss'].avg
        logging(log_output, log_file)
    logging('Done training', log_file)
