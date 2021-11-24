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


def download_all_models():
    subs = subs = ['academic-humanities', 'academic-stem', 'anime', 'astrology', 'conservative', 'hippie-spiritual', 'kpop', 'lgbtq', 'liberal', 'sports', 'tech-nerd']

    #coati: download the other 11 models here
    #coati: check if they're already downloaded before doing it again, and add a "setting up..." thing
    path_cwd = os.getcwd()
    path_models = 'static\\models'
    url = 'https://drive.google.com/uc?id=1-E3NJgfZbGY9b-EIho_hy_-62feeHTDn'
    output = os.path.join(path_cwd, path_models, 'nlpmodel3-academic-humanities.pkl')
    #cwd is just the first pytorch-django folder...

    try:
        gdown.download(url, str(output), quiet=False)
        return 'success!'
    except Exception as e:
        return e