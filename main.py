from __future__ import print_function
import gensim
import argparse
import torch
import pickle
import numpy as np
import pandas as pd
import os
import math
import random
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import data
# import scipy.io
from scipy.io import loadmat
from torch import nn, optim
from torch.nn import functional as F
from pathlib import Path
from gensim.models.fasttext import FastText as FT_gensim
import tracemalloc
from sklearn.model_selection import train_test_split
from eatm import EATM
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--data_path', type=str, default='data/20ng', help='directory containing data')
parser.add_argument('--emb_path', type=str, default='data/20ng_embeddings.txt', help='directory containing word embeddings')
parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')

### model-related arguments
parser.add_argument('--num_topics', type=int, default=100, help='number of topics')  # K
parser.add_argument('--rho_size', type=int, default=256, help='dimension of rho')  # V
parser.add_argument('--emb_size', type=int, default=256, help='dimension of embeddings')  # L
parser.add_argument('--t_hidden_size', type=int, default=1024, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--train_embeddings', type=int, default=0, help='whether to fix rho or train it')

### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train...150 for 20ng 100 for others')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')  # ?
parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=2, help='when to log training')
parser.add_argument('--visualize_every', type=int, default=10, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=100, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

## get data
# 1. vocabulary
wfolder = './data/preprocess'
dictionary = gensim.corpora.Dictionary()
dictionary = dictionary.load(os.path.join(wfolder, 'dictionary'))
training_set = np.load(os.path.join(wfolder, 'corpus_tfidf_train.npy'))
training_set = np.delete(training_set, -1, axis=0)
input_x_file = pd.read_csv(os.path.join(wfolder, 'corpus_train_authors_onehot.csv'))
input_x = np.array(input_x_file)
input_x = np.nan_to_num(input_x)
testing_set = np.load(os.path.join(wfolder, 'corpus_tfidf_test.npy'))
testing_set = np.delete(testing_set, -1, axis=0)
testing_author = pd.read_csv(os.path.join(wfolder, 'corpus_test_authors_onehot.csv'))
test_author = np.array(testing_author)
test_author = np.nan_to_num(test_author)
testing_set, valid, test_author, valid_author = train_test_split(testing_set, test_author, test_size=0.4,
                                                                 random_state=1)
test_1, test_2, test_a1, test_a2 = train_test_split(testing_set, test_author, test_size=0.5, random_state=1)

authors = pd.read_csv(os.path.join(wfolder, 'author_ids.map'))
authors_name = authors['name']

vocab = dictionary
vocab_size = len(vocab)
args.vocab_size = vocab_size

# 1. training data
args.num_docs_train = training_set.shape[0]

# 2. dev set

args.num_docs_valid = valid.shape[0]
# 3. test data

args.num_docs_test = test_1.shape[0] + test_2.shape[0]

args.num_docs_test_1 = test_1.shape[0]
args.num_docs_test_2 = test_2.shape[0]

x_num_classes = input_x.shape[1]

embeddings = None
if not args.train_embeddings:
    embeddings = data.read_embedding_matrix(vocab, device, load_trainned=False)
    args.embeddings_dim = embeddings.size()

print('=*' * 100)
print('Training an Embedded Topic Model on {} with the following settings: {}'.format(args.dataset.upper(), args))
print('=*' * 100)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

ckpt = Path.cwd().joinpath(args.save_path,
                           'EATM_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
                               args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip,
                               args.theta_act,
                               args.lr, args.batch_size, args.rho_size, args.train_embeddings))

## define model and optimizer
model = EATM(args.num_topics,
             vocab_size,
             args.t_hidden_size,
             args.rho_size,
             args.emb_size,
             x_num_classes,
             args.theta_act,
             embeddings,
             args.train_embeddings,
             args.enc_drop).to(device)

print('model: {}'.format(model))

optimizer = model.get_optimizer(args)

tracemalloc.start()
if args.mode == 'train':
    ## train model on data
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    print('\n')
    print('Visualizing model quality before training...', args.epochs)
    print('\n')
    for epoch in range(0, args.epochs):
        print("I am training for epoch", epoch)
        val_ppl = model.train_for_epoch(epoch, args, training_set, input_x)
        print("The validation scores", val_ppl)
        '''
        with open(ckpt, 'wb') as f:
            torch.save(model, f)
        '''
        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (
                    len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor

        '''
        if epoch % args.visualize_every == 0:
            model.visualize(args, vocabulary=vocab)
        '''
        all_val_ppls.append(val_ppl)
else:
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        test_ppl = model.evaluate(args, 'val', training_set, input_x, vocab, test_1, test_2, test_a1, test_a2, tc=1,
                                  td=1)
        ## get most used topics
        indices = torch.tensor(range(args.num_docs_train))
        indices = torch.split(indices, args.batch_size)
        thetaAvg = torch.zeros(1, args.num_topics).to(device)
        theta_weighted_average = torch.zeros(1, args.num_topics).to(device)
        cnt = 0
        for idx, indice in enumerate(indices):
            data_batch = data.get_batch(training_set, indice, device)
            sums = data_batch.sum(1).unsqueeze(1)
            cnt += sums.sum(0).squeeze().cpu().numpy()
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
        ## load embeddings
        try:
            rho_emb = model.rho.weight.cpu()
        except:
            rho_emb = model.rho.cpu()
        print(f'rho_etm_shape:{rho_emb.shape}')

        try:
            topics_emb = model.alphas.weight.cpu()
        except:
            topics_emb = model.alphas.cpu()
        print(f'rho_etm_shape:{topics_emb.shape}')

        try:
            authors_emb = model.author_alphas.weight.cpu()
        except:
            authors_emb = model.author_alphas.cpu()
        print(f'rho_etm_shape:{authors_emb.shape}')

        ## show topics
        beta = model.get_beta()

        print(f'beta_shape:{beta.shape}')
        topic_indices = list(np.random.choice(args.num_topics, 10))  # 10 random topics
        vis_list = []
        print('\n')
        for k in range(args.num_topics):  # topic_indices:
            gamma = beta[k]
            top_words = list(gamma.cpu().numpy().argsort()[-args.num_words:][::-1])
            topic_words = [vocab[a] for a in top_words]
            print('Topic {}: {}'.format(k, topic_words))
        if args.train_embeddings:
            ## show etm embeddings

            print('\n')
            print('ETM embeddings...')
            print('\n')

        ## show authors
        phi = model.get_phi()
        num_authors_show = 10
        author_count = input_x.T.sum(1)
        author_sidx = author_count.argsort()[-num_authors_show:][::-1]
        print(f'phi_shape:{phi.shape}')
        ## author_indices = list(np.random.choice(num_authors_show, 10))  # 10 random authors
        ## vis_list = []
        print('\n')
        for k in author_sidx:  # author_indices:
            gamma = phi[k]
            top_topics = list(gamma.cpu().numpy().argsort()[-5:][::-1])
            ## author_topics = [vocab[a] for a in top_words]
            print('Author {}: Topic {}'.format(authors_name[k], top_topics))

        ## save parameters
        np.save(os.path.join(wfolder, 'rho_emb.npy'), rho_emb.numpy())
        np.save(os.path.join(wfolder, 'topics_emb.npy'), topics_emb.numpy())
        np.save(os.path.join(wfolder, 'authors_emb.npy'), authors_emb.numpy())

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
tracemalloc.stop()
