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

import pathlib
posixpath_temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


path_cwd = os.getcwd()
path_df = 'static\\dataframes'
path_dls = 'static\\dataloaders'
path_models = 'static\\models'
path_nums200 = 'static\\nums200'
path_toks200 = 'static\\toks200'

def download_things(_url, _folder, _filename):
    path_cwd = os.getcwd()
    output = os.path.join(path_cwd, 'static', _folder, _filename)
    #cwd is just the first pytorch-django folder...

    try:
        gdown.download(_url, str(output), quiet=False)
        return 'success!'
    except Exception as e:
        return e

def download_all_models():
    #coati: download the other 11 models here
    #coati: check if they're already downloaded before doing it again, and add a "setting up..." thing

    subs = subs = ['academic-humanities', 'academic-stem', 'anime', 'astrology', 'conservative', 'hippie-spiritual', 'kpop', 'lgbtq', 'liberal', 'sports', 'tech-nerd']

    url = 'https://drive.google.com/uc?id=1-E3NJgfZbGY9b-EIho_hy_-62feeHTDn'
    folder = 'models'
    filename = 'nlpmodel3-academic-humanities.pkl'
    if not os.path.exists(os.path.join(path_cwd, path_models, filename)):
        download_things(url, folder, filename)

def download_toks200():
    url = 'https://drive.google.com/uc?id=1fx1HDjJ7O9Hryq6yu_AymzM5GtULcSWx'
    folder = 'toks200'
    filename = 'toks200-tweets.pkl'
    if not os.path.exists(os.path.join(path_cwd, path_toks200, filename)):
        download_things(url, folder, filename)


def download_nums200():
    url = 'https://drive.google.com/uc?id=1IgIcw_CRJQgdTo-Nn4sxk_g5Vd3xqiob'
    folder = 'nums200'
    filename = 'nums200-eachsub.pkl'
    if not os.path.exists(os.path.join(path_cwd, path_nums200, filename)):
        download_things(url, folder, filename)
