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
    df_eachsub = []

    try:
        df_eachsub = torch.load(os.path.join(path_cwd, path_df, 'df_eachsub_tweets.pkl'))
        return str(len(df_eachsub))
    except Exception as e:
        return e



    # for i in range(0, len(subs)):
    #     df = None
    #     newfolder = os.path.join(path_df, subs[i])
    #     num = 0
    #     docexists = True
    #     while docexists == True:
    #         num += 1
    #         newpath = os.path.join(newfolder, ('tweets_extracted_' + '{:03d}'.format(num) + '.txt'))
    #         if newpath.exists():
    #         print('Reading ' + subs[i] + ' ' + str(num))
    #         if df is None:
    #             df = pd.read_csv(newpath, sep='\n', header=None)
    #             df.columns = ['Tweet']
    #         else:
    #             df_new = pd.read_csv(newpath, sep='\n', header=None)
    #             df_new.columns = ['Tweet']
    #             df = df.append(df_new)
    #         else:
    #         docexists = False
    #         df_eachsub.append(df)
    #         print(df.shape)