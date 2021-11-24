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
    path_models = 'static\\models'
    





    #coati: go through and retool all this
    path = Path("/content/gdrive/MyDrive/fastai_datasets/tweets-by-subculture")
    subs = ['academic-humanities', 'academic-stem', 'anime', 'astrology', 'conservative', 'hippie-spiritual', 'kpop', 'lgbtq', 'liberal', 'sports', 'tech-nerd']
    df_eachsub = []
    for i in range(0, len(subs)):
    df = None
    newfolder = path/subs[i]
    num = 0
    docexists = True
    while docexists == True:
        num += 1
        newpath = Path(newfolder/('tweets_extracted_' + '{:03d}'.format(num) + '.txt'))
        if newpath.exists():
        print('Reading ' + subs[i] + ' ' + str(num))
        if df is None:
            df = pd.read_csv(newpath, sep='\n', header=None)
            df.columns = ['Tweet']
        else:
            df_new = pd.read_csv(newpath, sep='\n', header=None)
            df_new.columns = ['Tweet']
            df = df.append(df_new)
        else:
        docexists = False
        df_eachsub.append(df)
        print(df.shape)