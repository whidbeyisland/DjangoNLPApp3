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
        # pred = self.insert_rare_words(pred, rare_words)
        pred = self.check_grammar(pred)
        pred = self.alter_capitalization(pred)
        pred = self.truncate_tail(pred)
        pred = self.alter_punctuation(pred)
        return pred
    
    def check_grammar(self, pred):
        pred = pred.strip()

        pred = re.sub('’', '\'', pred)

        pred = re.sub(' \'', '\'', pred)
        pred = re.sub(' /', '/', pred)
        pred = re.sub('/ ', '/', pred)
        pred = re.sub(' :', ':', pred)
        pred = re.sub('\( ', '(', pred)
        pred = re.sub(' \)', ')', pred)
        pred = re.sub(' \.', '.', pred)
        pred = re.sub(' \,', ',', pred)
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
    
    def truncate_tail(self, pred):
        pred = re.sub('\.[^\.]*$', '', pred)
        return pred
    
    # currently not using this, "synsets" not a high-quality synonym database, but may try to make this
    # method usable later
    def insert_rare_words(self, pred, rare_words):
        # rare_words.append('banana')
        # rare_words.append('fox')
        random.shuffle(rare_words)
        threshold = 0.5
        replaced_so_far = 0
        max_replace_count = 5

        # for each of the person's rare words, find the word in the sentence that's the most similar to it
        # and above the threshold (if such a word exists), and then replace it with the rare word
        words = pred.split()
        words_to_replace = []
        for i in range(0, len(rare_words)):
            if replaced_so_far < max_replace_count:
                best_word_simil = 0
                
                try:
                    syn1 = wordnet.synsets(rare_words[i])[0]
                    for word in words:
                        try:
                            syn2 = wordnet.synsets(word)[0]
                            simil = syn1.wup_similarity(syn2)
                            if simil is not None:
                                ind = words.index(word)
                                # if you've found a new best word in the sentence
                                if simil > threshold and simil > best_word_simil:
                                    best_word_simil = simil
                                    replaced_so_far += 1
                                    if best_word_simil > 0:
                                        words_to_replace.append([ind, i])
                        except Exception as e:
                            pass # print('2 ' + str(e))
                except Exception as e:
                    pass # print('1 ' + str(e))
        if len(words_to_replace) > 0:
            for pair in words_to_replace:
                words[pair[1]] = rare_words[pair[1]]
        pred = ' '.join(words)

        return pred