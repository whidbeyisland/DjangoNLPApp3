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
nltk.download('brown')
from nltk.corpus import brown
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

    # coati: create intros like "X is just...", "I love X because..."
    def intro_from_prompt(self, topic):
        topic = topic.strip()

        index = 1 if self.is_plural(topic) else 0
        rand = random.randint(0, 100)
        intro = self.pick_stub(topic, rand, index)

        return intro
    
    def pick_stub(self, topic, rand, index):
        if 0 <= rand < 10:
            stub = 'I really like ' + topic
        elif 10 <= rand < 20:
            stub = 'I love ' + topic
        elif 20 <= rand < 30:
            stub = 'I just love ' + topic
        elif 30 <= rand < 40:
            if index == 0:
                stub = topic + ' is'
            else:
                stub = topic + ' are'
        elif 40 <= rand < 50:
            if index == 0:
                stub = topic + ' is just'
            else:
                stub = topic + ' are just'
        elif 50 <= rand < 60:
            if index == 0:
                stub = topic + ' is absolutely'
            else:
                stub = topic + ' are absolutely'
        elif 60 <= rand < 70:
            if index == 0:
                stub = topic + ' makes me want to'
            else:
                stub = topic + ' make me want to'
        elif 70 <= rand < 80:
            if index == 0:
                stub = topic + ' makes me think that'
            else:
                stub = topic + ' make me think that'
        elif 80 <= rand < 90:
            if index == 0:
                stub = topic + ' really makes me want to'
            else:
                stub = topic + ' really make me want to'
        else:
            if index == 0:
                stub = topic + ' really makes me think that'
            else:
                stub = topic + ' really make me think that'

        if 0 <= rand < 30:
            if 6 <= rand % 10 < 8:
                stub = stub + ' because'
            elif 8 <= rand % 10 < 10:
                stub = stub + ' since'
        
        if rand % 9 == 0:
            stub = 'Oh my god, ' + stub

        return stub

    def is_plural(self, topic):
        # coati: not perfect but just covering most common irregular plurals, maybe find an Excel of thousands of them
        irregular_plurals = [
            'men', 'women', 'children', 'people'
            'deer', 'buffalo', 'fish', 'bison', 'buffalo', 'oxen', 'cattle', 'moose', 'lice', 'geese',
            'dice',
            'you', 'u', 'they', 'them', 'y\'all', 'yall', 'youse', 'yinz', 'ppl'
        ]
        irregular_plural_endings_2 = [
            'ia', 'ii'
        ]
        irregular_plural_endings_3 = [
            'men', 'ata',
            'ppl'
        ]
        irregular_plural_endings_4 = [
            'fish'
        ]
        irregular_plural_endings_6 = [
            'people'
        ]
        if topic[-1] == 's' or topic in irregular_plurals or topic[-2:] in irregular_plural_endings_2 or topic[-3:] in irregular_plural_endings_3 or topic[-4:] in irregular_plural_endings_4 or topic[-6:] in irregular_plural_endings_6:
            return True
        else:
            return False

    def apply_manipulations(self, pred, topic, user_styles, rare_words):
        # pred = self.insert_rare_words(pred, rare_words)
        pred = self.check_grammar(pred)
        pred = pred.replace('xxunk', topic)
        pred = self.alter_capitalization(pred, user_styles)
        pred = self.truncate_tail(pred)
        pred = self.alter_punctuation(pred, user_styles)
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

        # coati: has a tendency to insert "n't" where it shouldn't be
        valid_neg_modals = [
            'isn\'t', 'aren\'t', 'ain\'t', 'doesn\'t', 'don\'t', 'can\'t', 'couldn\'t', 'won\'t', 'wouldn\'t',
            'shan\'t', 'shouldn\'t', 'mayn\'t', 'mightn\'t'
        ]
        pred_words = pred.split()
        for i in range (0, len(pred_words)):
            pred_word = pred_words[i]
            if pred_word[-3:] == 'n\'t':
                if pred_word not in valid_neg_modals:
                    pred_word = pred_word[:-3]
                    pred_words[i] = pred_word
        pred = ' '.join(pred_words)
        
        # pred = re.sub(r' (^(do|does|is|are|ai|ca|wo|would|might))n\'t', r' \1', pred)

        pred = pred.rstrip(punctuation)
        return pred
    
    def alter_capitalization(self, pred, user_styles):
        rand = random.randint(0, 10)
        if rand < user_styles[0]:
            return pred.capitalize()
        else:
            return pred.lower()
    
    def alter_punctuation(self, pred, user_styles):
        rand = random.randint(0, 10)
        rand2 = random.randint(0, 5)
        possible_punctuation = ['.', '.', '.', '?', '!']
        if rand < user_styles[1]:
            return pred + possible_punctuation[rand2]
        else:
            return pred
    
    def truncate_tail(self, pred):
        pred = re.sub('\.[^\.]*$', '', pred)
        return pred
    
    def find_syns(self, queried_word):
        all_syns = []
        likely_proper_name = False
        try:
            syn1 = wordnet.synsets(queried_word)
            for i in range(0, len(syn1)):
                try:
                    lemmata = syn1[i].lemma_names()
                    for j in range(0, len(lemmata)):
                        # if the word's synonyms include any proper nouns, the word itself is probably a
                        # proper noun, and we shouldn't find synonyms for it
                        if re.match('^[A-Z]', lemmata[j]):
                            likely_proper_name = True
                        all_syns.append(lemmata[j])
                except Exception as e:
                    pass
        except Exception as e:
            print(e)
        all_syns = filter(lambda word: word != queried_word, all_syns)
        all_syns = list(set(all_syns))

        if likely_proper_name == True:
            all_syns = []

        return all_syns    

    
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
