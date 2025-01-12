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

    def intro_from_prompt(self, topic, rare_words):
        topic = topic.strip()
        intro = topic + ' is' # failsafe in case of exception

        # some variables for generating random intros
        index = 1 if self.is_plural(topic) else 0
        rand = random.randint(0, 99)
        rand_use_rare_word = random.randint(0, 9)
        use_rare_word_threshold = 5

        most_applicable_rare_words = self.find_closest_rare_words(topic, rare_words)
        # if any of the person's rare words were similar to the topic, then maybe use them to generate an intro
        if len(most_applicable_rare_words) > 0:
            if rand_use_rare_word >= use_rare_word_threshold:
                rand_rare_word = random.randint(0, len(most_applicable_rare_words) - 1)
                print('index ' + str(rand_rare_word) + ' out of ' + str(len(most_applicable_rare_words)))
                rare_word = most_applicable_rare_words[rand_rare_word]
                intro = self.pick_stub_with_rare_word(topic, index, rare_word, rand)
            else:
                intro = self.pick_stub(topic, rand, index)
        # otherwise, only use the topic
        else:
            intro = self.pick_stub(topic, rand, index)

        return intro

    def pick_stub_with_rare_word(self, topic, index, rare_word, rand):
        if rare_word[1] == 'NNP': # proper noun, singular
            if 0 <= rand < 25:
                stub = 'I really like ' + rare_word[0] + ' for ' + topic
            elif 25 <= rand < 50:
                stub = 'I love ' + rare_word[0] + ' for ' + topic
            elif 50 <= rand < 75:
                stub = rare_word[0] + ' for ' + topic + ' is just'
            elif rand >= 75:
                stub = rare_word[0] + ' for ' + topic + ' really makes me think that'
        elif rare_word[1] == 'NN': # common noun, singular
            if 0 <= rand < 25:
                stub = 'I really like the ' + rare_word[0] + ' for ' + topic
            elif 25 <= rand < 50:
                stub = 'I love the ' + rare_word[0] + ' for ' + topic
            elif 50 <= rand < 75:
                stub = 'the ' + rare_word[0] + ' for ' + topic + ' is just'
            elif rand >= 75:
                stub = 'the ' + rare_word[0] + ' for ' + topic + ' really makes me think that'
        elif rare_word[1] == 'NNS': # plural noun
            if 0 <= rand < 25:
                stub = 'I really like ' + rare_word[0] + ' for ' + topic
            elif 25 <= rand < 50:
                stub = 'I love ' + rare_word[0] + ' for ' + topic
            elif 50 <= rand < 75:
                stub = rare_word[0] + ' for ' + topic + ' are just'
            elif rand >= 75:
                stub = rare_word[0] + ' for ' + topic + ' really make me think that'
        elif rare_word[1] == 'JJ': # adjective
            if 0 <= rand < 16:
                stub = 'I really like the ' + rare_word[0] + ' ' + topic
            elif 16 <= rand < 33:
                stub = 'I love the ' + rare_word[0] + ' ' + topic
            elif 33 <= rand < 50:
                if index == 0:
                    stub = 'the ' + rare_word[0] + ' ' + topic + ' is just'
                else:
                    stub = 'the ' + rare_word[0] + ' ' + topic + ' are just'
            elif 50 <= rand < 66:
                if index == 0:
                    stub = 'the ' + rare_word[0] + ' ' + topic + ' really makes me think that'
                else:
                    stub = 'the ' + rare_word[0] + ' ' + topic + ' really make me think that'
            elif 66 <= rand < 83:
                stub = 'being ' + rare_word[0] + ' for ' + topic + 'is'
            elif rand >= 83:
                stub = 'being ' + rare_word[0] + ' for ' + topic + 'is just'
        elif rare_word[1][:2] == 'VB': # verb, any tense/aspect
            # coati: make separate paths for different tenses/aspects. can you inflect non-present verbs to present?
            if 0 <= rand < 50:
                if index == 0:
                    stub = topic + ' makes me want to ' + rare_word[0]
                else:
                    stub = topic + ' make me want to ' + rare_word[0]
            else:
                if index == 0:
                    stub = topic + ' really makes me want to ' + rare_word[0]
                else:
                    stub = topic + ' really make me want to ' + rare_word[0]
        elif rare_word[1][:2] == 'RB': # adverb
            if 0 <= rand < 50:
                if index == 0:
                    stub = topic + ' is ' + rare_word[0]
                else:
                    stub = topic + ' are ' + rare_word[0]
            else:
                if index == 0:
                    stub = rare_word[0] + ', ' + topic + ' is'
                else:
                    stub = rare_word[0] + ', ' + topic + ' are'
        else:
            if 0 <= rand < 25:
                stub = 'I really like ' + rare_word[0] + ' for ' + topic
            elif 25 <= rand < 50:
                stub = 'I love ' + rare_word[0] + ' for ' + topic
            elif 50 <= rand < 75:
                stub = rare_word[0] + ' for ' + topic + ' is just'
            elif rand >= 75:
                stub = rare_word[0] + ' for ' + topic + ' really makes me think that'

        return stub
    
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

        # making sure "n't" not inserted where it shouldn't be --- model has a tendency to do that
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

        pred = pred.rstrip(punctuation)
        return pred
    
    def alter_capitalization(self, pred, user_styles):
        # coati: split pred into sentences somehow but keep their original punctuation, probably with a separate
        # method
        rand = random.randint(0, 9)
        # pred = re.split('\. |\? |! ', pred)
        if rand < user_styles[0]:
            return pred.capitalize()
        else:
            return pred.lower()
    
    def alter_punctuation(self, pred, user_styles):
        rand = random.randint(0, 9)
        rand2 = random.randint(0, 4)
        possible_punctuation = ['.', '.', '.', '?', '!']
        if rand < user_styles[1]:
            return pred + possible_punctuation[rand2]
        else:
            return pred
    
    def truncate_tail(self, pred):
        pred = re.sub('\.[^\.]*$', '', pred)
        return pred

    def find_closest_rare_words(self, topic, rare_words):
        all_close_rare_words = []
        try:
            # look at top 3 definitions of the topic separately --- the topic is just provided as a word,
            # so you can't tell what POS the user meant it as (e.g. "abstract" could have been meant as a
            # noun, verb, or adjective)
            syn1 = wordnet.synsets(topic)
            for i in range(0, len(syn1)):
                if i < 3:
                    syn1_this_def = syn1[i]
                    for j in range(0, len(rare_words)):
                        # print(rare_words[j][0])
                        try:
                            # look at top 3 definitions of the rare word separately
                            syn2 = wordnet.synsets(rare_words[j][0])
                            for k in range(0, len(syn2)):
                                if k < 3:
                                    syn2_this_def = syn2[k]
                                    simil = syn1_this_def.wup_similarity(syn2_this_def)
                                    if simil is not None:
                                        if simil > 0.4:
                                            print('simil.......... ' + str(simil))
                                            if rare_words[j] not in all_close_rare_words:
                                                all_close_rare_words.append(rare_words[j])
                                            # print(len(all_close_rare_words))
                                    # if k.pos == 'n' and rare_words[j][1][0] == 'N':
                                    #     pass
                                    # elif k.pos == 'v' and rare_words[j][1][0:2] == 'VB':
                                    #     pass
                                    # elif k.pos == 'adj' and rare_words[j][1][0:2] == 'JJ':
                                    #     pass
                                    # else:
                                    #     pass
                        except:
                            pass
        except:
            pass
        return all_close_rare_words
    
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
