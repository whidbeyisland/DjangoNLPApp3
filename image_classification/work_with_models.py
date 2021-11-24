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
    
