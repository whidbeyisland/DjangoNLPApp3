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

from torchvision import models
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.conf import settings
#from fastbook import *
from torchtext.data import get_tokenizer
from fastai.text.all import *
#from pathlib import Path

nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk import FreqDist
from string import punctuation

from .forms import TextEntryForm
from .download_pkls import *

import pathlib
posixpath_temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



subs = ['academic-humanities', 'academic-stem', 'anime', 'astrology', 'conservative', 'hippie-spiritual', 'kpop', 'lgbtq', 'liberal', 'sports', 'tech-nerd']
path_cwd = os.getcwd()
path_df = 'static\\dataframes'
path_dls = 'static\\dataloaders'
path_models = 'static\\models'
path_nums200 = 'static\\nums200'
path_toks200 = 'static\\toks200'
path_tweets = 'static\\tweets-by-user'
max_tweets = 300
tweets_to_analyze = 100

def get_tweets(df):
    return L(df.iloc[i, 0] for i in range(0, df.shape[0]))

def subword(sz):
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])

class WorkWithModels: 
    d = None

    df_eachsub = []
    tkn_eachsub = []
    txts_eachsub = []
    toks200_eachsub = []
    nums200_eachsub = []
    dls_eachsub = []
    learn = None

    df_c = None
    tkn_c = None
    toks200_c = None
    nums200_c = None
    dls_c = None
    learn_c = None

    def __init__(self, d):
        self.d = d
    
    def get_categorization_assets_ready(self):
        print('Getting assets for categorization, hang tight................')
        print('Loading dataframes...')
        try:
            self.df_c = torch.load(os.path.join(path_cwd, path_df, 'df_categorize.pkl'))
        except Exception as e:
            print(e)

        spacy = WordTokenizer()
        tkn = Tokenizer(spacy)
        self.tkn_c = tkn

        print('Loading txts...')
        self.txts_c = L(self.df_c.iloc[i, 0] for i in range(0, self.df_c.shape[0]))

        print('Loading toks200...')
        try:
            #COATI: host this online so you can download it, currently it has no way of getting to the project
            self.toks200_c = torch.load(os.path.join(path_cwd, path_toks200, 'toks200_c.pkl'))
        except Exception as e:
            print(e)

        print('Loading nums200...')
        try:
            #COATI: host this online so you can download it, currently it has no way of getting to the project
            self.nums200_c = torch.load(os.path.join(path_cwd, path_nums200, 'nums200_c.pkl'))
        except Exception as e:
            print(e)

        print('Loading dataloaders...')
        try:
            self.d.download_dls_c()
            self.dls_c = torch.load(os.path.join(path_cwd, path_dls, 'dls-nlp-c.pkl'))
        except Exception as e:
            print(e)
        
        print('Loading learners...')
        try:
            self.d.download_learn_c()
            self.learn_c = torch.load(os.path.join(path_cwd, path_models, 'nlpmodel3_c.pkl'))
        except Exception as e:
            print(e)
        
        print('Success!')

    def get_generation_assets_ready(self):
        print(self.d.testfunc())
        print('Getting assets for tweet generation, hang tight................')
        print('Loading dataframes...')
        try:
            self.df_eachsub = torch.load(os.path.join(path_cwd, path_df, 'df_eachsub_tweets.pkl'))
            print(str(len(self.df_eachsub)))
        except Exception as e:
            print(e)
        
        spacy = WordTokenizer()
        for i in range(0, len(subs)):
            tkn = Tokenizer(spacy)
            self.tkn_eachsub.append(tkn)
        
        print('Loading txts...')
        try:
            self.txts_eachsub = torch.load(os.path.join(path_cwd, path_df, 'txts_eachsub.pkl'))
            print(str(len(self.txts_eachsub)))
        except Exception as e:
            print(e)
        #coati: TO DO................ store txts_eachsub.pkl on drive so you can download it, currently the program has no way
        #of creating it

        print('Loading toks200...')
        try:
            self.d.download_toks200()
            self.toks200_eachsub = torch.load(os.path.join(path_cwd, path_toks200, 'toks200-tweets.pkl'))
            print(str(len(self.toks200_eachsub)))
        except Exception as e:
            print(e)

        print('Loading nums200...')
        try:
            self.d.download_nums200()
            self.nums200_eachsub = torch.load(os.path.join(path_cwd, path_nums200, 'nums200-eachsub.pkl'))
            print(str(len(self.nums200_eachsub)))
        except Exception as e:
            print(e)
        
        print('Loading dataloaders...')
        try:
            for i in range(0, len(subs)):
                filename = 'dls-nlp-' + subs[i] + '-ALT.pkl'
                dls_thissub = torch.load(os.path.join(path_cwd, path_dls, filename))
                self.dls_eachsub.append(dls_thissub)
            print(str(len(self.dls_eachsub)))
        except Exception as e:
            print(e)
        
        #coati: for now just doing 1 learner, but will eventually do all of them

        print('Loading learners...')
        #learn_eachsub = []
        try:
            filename = 'nlpmodel3-academic-humanities.pkl'
            self.learn = torch.load(os.path.join(path_cwd, path_models, filename))
            #, map_location=torch.device('cpu')
            #for i in range(0, len(subs)):
            #    filename = 'dls-nlp-' + subs[i] + '-ALT.pkl'
            #    dls_thissub = torch.load(os.path.join(path_cwd, path_dls, filename))
            #    dls_eachsub.append(dls_thissub)
            print('Loaded')
        except Exception as e:
            print(e)
        
    def download_user_tweets(self, username):
        print('Downloading tweets by user ' + username + '...')

        tweets_list = []

        for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + username).get_items()): #declare a username 
            if i > max_tweets:
                break
            tweets_list.append([tweet.content])
            #racc: tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
 
        tweets_df = pd.DataFrame(tweets_list, columns=['Content'])
        #print(tweets_df.iloc[0:10,:])
        tweets_df.to_csv(os.path.join(path_tweets, 'tweets-' + username + '.csv'))

        # scraper = snscrape.modules.twitter.TwitterUserScraper('textfiles')
        # for tweet in scraper.get_items():
        #     print(tweet.user.username)
        #     print(tweet.date)
        #     print(tweet.content)

        #os.system('snscrape --max-results ' + str(max_tweets) + ' twitter-user ' + username + ' >tweets-by-user-' + username + '.txt')

    def categorize_user(self, username):
        try:
            self.download_user_tweets(username)
            df_user = pd.read_csv(os.path.join(path_tweets, 'tweets-' + username + '.csv'), error_bad_lines=False)
            print(df_user.shape)
        except Exception as e:
            print(e)
        
        first100tweets = df_user.loc[:100, 'Content']
        #first100tweets = df_user.iloc[0:tweets_to_analyze,0]
        print(first100tweets.shape)
        print(first100tweets[0])

        #predictions of subculture for each tweet
        preds_each_tweet = []
        for i in range(0, len(first100tweets)):
            print('Analyzing tweet #', i)
            preds_this_tweet = self.learn_c.predict(first100tweets[i])
            print(preds_this_tweet)
            #coati: leaving off here, figure out why this is just printing the tweets instead of the preds
            preds_each_tweet.append(preds_this_tweet)
        all_preds_each_categ = torch.stack(preds_each_tweet)

        df_preds = pd.DataFrame(all_preds_each_categ.numpy())
        df_preds.columns = subs

        #predictions of subculture for all of the user's tweets
        preds_overall_each_categ = []
        for i in range(0, len(subs)):
            pred_overall_this_categ = df_preds[subs[i]].mean()
            preds_overall_each_categ.append(pred_overall_this_categ)

        #sorting these overall predictions from most to least likely
        preds_overall_each_categ_sorted = copy.deepcopy(preds_overall_each_categ)
        preds_overall_each_categ_sorted.sort(reverse=True)

        for i in range(0, len(preds_overall_each_categ_sorted)):
            cur_pred = preds_overall_each_categ_sorted[i]
            orig_index_of_pred = preds_overall_each_categ.index(cur_pred)
            print('Likelihood of being in group ' + subs[orig_index_of_pred] + ': ' + str(cur_pred))

    def get_tweet_prediction(self, username, topic):
        TEXT = topic
        N_WORDS = 40
        N_SENTENCES = 4
        preds = [self.learn.predict(TEXT, N_WORDS, temperature=0.75) 
                for _ in range(N_SENTENCES)]
        print('-------------------------------------------')
        print("\n".join(preds))
        print('-------------------------------------------')
    
#return 'got to end'