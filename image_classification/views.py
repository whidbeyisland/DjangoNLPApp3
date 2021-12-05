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
import datetime

from .forms import TextEntryForm
from .download_pkls import *
from .work_with_models import *
from .tweet_manipulations import *



def index(request):
    predicted_label = None
    today = datetime.now()
    todaydate = today.strftime("%I:%M %p · %B %d, %Y")
    user_alias = 'Username Alias'
    username = 'username'
    predicted_tweet = 'Tweet goes here'

    if request.method == 'POST':
        form = TextEntryForm(request.POST, request.FILES)
        if form.is_valid():
            username = form.cleaned_data['username']
            prompt = form.cleaned_data['prompt']
            print('TEEEEEEEEEEEEEEEEEST...1 ' + prompt)

            try:
                d = DownloadPkls()
                t = TweetManipulations()
                w = WorkWithModels(d, t)

                
                w.download_user_tweets(username)
                w.get_user_assets_ready(username)
                w.get_rare_words(username)

                # w.get_categorization_assets_ready()
                w.get_generation_assets_ready()
                # subs_to_generate = w.categorize_user(username)
                w.subs_eachuser[username] = [0, 1, 2]
                w.get_tweet_prediction(username, prompt)
                # w.get_tweet_prediction(username, 'People from ancient Mesopotamia')
                # w.get_tweet_prediction(username, 'Japan is a nation')
                # w.get_tweet_prediction(username, 'Homophobia')
                # w.get_tweet_prediction(username, 'It is highly disappointing that')
                w.get_tweet_prediction(username, 'I really don\'t like')
                w.get_tweet_prediction(username, 'My absolute favorite')
                predicted_tweet = 'Tweet goes here'
                predicted_label = 'success!'
                user_alias = username + 'Alias' # coati: retrieve person's alias

            except RuntimeError as re:
                predicted_label = re
                # predicted_label = "Prediction Error"

    else:
        form = TextEntryForm()

    context = {
        'form': form,
        'predicted_tweet': predicted_tweet,
        'predicted_label': predicted_label,
        'username': username,
        'user_alias': user_alias,
        'todaydate': todaydate
    }
    return render(request, 'image_classification/index.html', context)
