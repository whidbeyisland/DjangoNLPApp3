import base64
import io
import json
import os
import gdown
#import fastbook
#fastbook.setup_book()
import fastai
import pandas as pd
import requests
import torchtext
import nltk
import snscrape.modules.twitter as sntwitter
from copy import deepcopy

from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
#from fastbook import *
from torchtext.data import get_tokenizer
from fastai.text.all import *

nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import FreqDist
from string import punctuation

from .forms import TextEntryForm
from .download_pkls import *
from .work_with_models import *

import pathlib
posixpath_temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



class TweetManipulations:
    rare_words = []

    def __init__(self):
        pass

    def apply_manipulations(self, pred, rare_words):
        pred = self.insert_rare_words(pred)
        pred = self.check_grammar(pred)
        pred = self.alter_capitalization(pred)
        pred = self.alter_punctuation(pred)
        return pred

    def insert_rare_words(self, pred):
        return pred
    
    def check_grammar(self, pred):
        pred = pred.strip()

        pred = re.sub('’', '\'')

        pred = re.sub(' \'', '\'', pred)
        pred = re.sub(' /', '/', pred)
        pred = re.sub('/ ', '/', pred)
        pred = re.sub(' :', ':', pred)
        pred = re.sub('\\( ', '(', pred)
        pred = re.sub(' \\)', ')', pred)
        pred = re.sub(' \\.', '.', pred)
        pred = re.sub(' \\,', ',', pred)
        pred = re.sub('=', '', pred)
        pred = re.sub('\n', ' ', pred)
        pred = re.sub('"', '', pred)
        pred = re.sub(' +', ' ', pred)
        pred = re.sub('^ +', '', pred)
        pred = re.sub(' \?', '?', pred)
        pred = re.sub(' !', '!', pred)
        
        pred = re.sub('i\'re', 'i\'m', pred)
        pred = re.sub('i\'s', 'i\'m', pred)
        pred = re.sub(' n\'t', 'n\'t', pred)
        pred = re.sub(' nt', 'nt', pred)

        pred = pred.rstrip(punctuation)
        return pred
    
    def alter_capitalization(self, pred):
        return pred.lower()
    
    def alter_punctuation(self, pred):
        return pred