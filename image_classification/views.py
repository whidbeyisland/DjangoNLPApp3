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
from .likes_replies_generator import LikesRepliesGenerator

def index(request):
    request_complete = False
    predicted_label = None
    today = datetime.now()
    todaydate = today.strftime("%I:%M %p · %B %d, %Y")
    user_alias = 'Username Alias'
    username = 'username'
    predicted_tweets = ['Tweet goes here', 'Tweet goes here']
    num_likes_replies = LikesRepliesGenerator().generate(2)
    num_likes_str_0, num_replies_str_0 = num_likes_replies[0][0], num_likes_replies[0][1]
    num_likes_str_1, num_replies_str_1 = num_likes_replies[1][0], num_likes_replies[1][1]

    if request.method == 'POST':
        form = TextEntryForm(request.POST, request.FILES)
        if form.is_valid():
            username = form.cleaned_data['username']
            prompt = form.cleaned_data['prompt']

            try:
                d = DownloadPkls()
                t = TweetManipulations()
                w = WorkWithModels(d, t)

                w.download_user_tweets(username)
                w.get_user_assets_ready(username)
                w.get_rare_words(username)
                w.get_generation_assets_ready()

                # w.get_categorization_assets_ready()
                # subs_to_generate = w.categorize_user(username)
                w.subs_eachuser[username] = [0, 1, 2]
                predicted_tweets = w.get_tweet_predictions(username, prompt)
                predicted_label = 'success!'
                user_alias = username # coati: retrieve person's alias
                request_complete = True

            except RuntimeError as re:
                predicted_label = re
                # predicted_label = "Prediction Error"

    else:
        form = TextEntryForm()

    context = {
        'form': form,
        'predicted_tweets': predicted_tweets,
        'predicted_label': predicted_label,
        'username': username,
        'user_alias': user_alias,
        'todaydate': todaydate,
        'num_likes_str_0': num_likes_str_0,
        'num_replies_str_0': num_replies_str_0,
        'num_likes_str_1': num_likes_str_1,
        'num_replies_str_1': num_replies_str_1,
        'request_complete': request_complete
    }
    return render(request, 'image_classification/index.html', context)
