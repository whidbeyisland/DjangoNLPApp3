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
    def __init__(self):
        pass

    def apply_manipulations(self, pred, rare_words):
        pred = self.insert_rare_words(pred, rare_words)
        pred = self.check_grammar(pred)
        pred = self.alter_capitalization(pred)
        pred = self.alter_punctuation(pred)
        return pred

    def insert_rare_words(self, pred, rare_words):
        threshold = 0.4

        rare_words.append('banana')
        rare_words.append('fox')

        # for each word in the prediction, find the person's rare word that is most similar to it and also over
        # the threshold, and change it to that rare word
        words = pred.split()
        words_to_replace = []
        for i in range(0, len(words)):
            best_rare_word = None
            best_rare_word_simil = 0
            
            try:
                syn1 = wordnet.synsets(words[i])[0]
                for rare_word in rare_words:
                    try:
                        syn2 = wordnet.synsets(rare_word)[0]
                        simil = syn1.wup_similarity(syn2)
                        if simil is not None:
                            print(rare_word)
                            if simil > threshold and simil > best_rare_word_simil:
                                best_rare_word = rare_word
                                best_rare_word_simil = simil
                            if best_rare_word_simil > 0:
                                words_to_replace.append([i, best_rare_word])
                    except Exception as e:
                        pass # print('2 ' + str(e))
            except Exception as e:
                pass # print('1 ' + str(e))
        if len(words_to_replace) > 0:
            for pair in words_to_replace:
                words[pair[0]] = pair[1]
        pred = ' '.join(words)

        return pred
    
    def check_grammar(self, pred):
        pred = pred.strip()

        pred = re.sub('’', '\'', pred)

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
        pred = re.sub(' -', '-', pred)
        pred = re.sub('- ', '-', pred)
        
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