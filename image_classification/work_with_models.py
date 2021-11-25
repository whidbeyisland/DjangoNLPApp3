import base64
import io
import json
import os
import gdown
import fastbook
fastbook.setup_book()
import fastai
import pandas as pd
import requests
import torchtext
import nltk

from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from fastbook import *
from torchtext.data import get_tokenizer
from fastai.text.all import *
#from pathlib import Path

nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import FreqDist
from string import punctuation

from .forms import ImageUploadForm


def get_tweets(df):
  return L(df.iloc[i, 0] for i in range(0, df.shape[0]))

def subword(sz):
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])

def get_tweet_prediction(account, topic):
    #coati: account will be person's name
    
    path_cwd = os.getcwd()
    path_df = 'static\\dataframes'
    path_dls = 'static\\dataloaders'
    path_models = 'static\\models'
    path_nums200 = 'static\\nums200'
    path_toks200 = 'static\\toks200'

    #coati: go through and retool all this
    subs = ['academic-humanities', 'academic-stem', 'anime', 'astrology', 'conservative', 'hippie-spiritual', 'kpop', 'lgbtq', 'liberal', 'sports', 'tech-nerd']
    
    print('Loading dataframes...')
    df_eachsub = []
    try:
        df_eachsub = torch.load(os.path.join(path_cwd, path_df, 'df_eachsub_tweets.pkl'))
        print(str(len(df_eachsub)))
    except Exception as e:
        print(e)
    
    spacy = WordTokenizer()
    tkn_eachsub = []
    for i in range(0, len(subs)):
        tkn = Tokenizer(spacy)
        tkn_eachsub.append(tkn)
    
    print('Loading txts...')
    txts_eachsub = []
    try:
        txts_eachsub = torch.load(os.path.join(pawh_cwd, path_df, 'txts_eachsub.pkl'))
        print(str(len(txts_eachsub)))
    except Exception as e:
        print(e)
    #racc: JUST RUN THIS PART ONCE, AFTER THE .PKL IS CREATED, REMOVE IT
    # for j in range(0, len(subs)):
    #     print(df_eachsub[j].shape[0])
    #     txts = L(df_eachsub[j].iloc[i, 0] for i in range(0, df_eachsub[j].shape[0]))
    #     txts_eachsub.append(txts)
    # torch.save(txts_eachsub, os.path.join(path_cwd, path_df, 'txts_eachsub.pkl'))




    #coati: VERIFIED UP TO HERE WORKS...




    print('Loading toks200...')
    toks200_eachsub = []
    try:
        download_toks200()
        toks200_eachsub = torch.load(os.path.join(path_cwd, path_toks200, 'toks200_tweets.pkl'))
        print(str(len(toks200_eachsub)))
    except Exception as e:
        print(e)
    
    print('Loading dataloaders...')
    dls_eachsub = []
    try:
        for i in range(0, len(subs)):
            filename = 'dls-nlp-' + subs[i] + '-ALT.pkl'
            dls_thissub = torch.load(os.path.join(pawh_cwd, path_dls, filename))
            dls_eachsub.append(dls_thissub)
        print(str(len(dls_eachsub)))
    except Exception as e:
        print(e)
    
    return 'got to end'