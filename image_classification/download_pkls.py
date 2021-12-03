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

from .forms import TextEntryForm
#from .work_with_models import *

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
urls_eachsub2 = {
    'academic-humanities': 'https://drive.google.com/uc?id=1-E3NJgfZbGY9b-EIho_hy_-62feeHTDn',
    'academic-stem': 'https://drive.google.com/uc?id=1--FLwQk8WLMsSPsr9A0NuHeJlXEwwjOL',
    'anime': 'https://drive.google.com/uc?id=1-2DtyFH5lY0i4Kr951BKflG-wirCzdxT',
    'astrology': 'https://drive.google.com/uc?id=102vZq6dnVG7gm5-yE8pvNflg2QqFmauz',
    'conservative': 'https://drive.google.com/uc?id=1-2On6tehSC6e0T0ottO8lWvfMCAe8gte',
    'hippie-spiritual': 'https://drive.google.com/uc?id=1-AsDJOOT4z7Mt3skW2ChN9xEhaSjbNf-',
    'kpop': 'https://drive.google.com/uc?id=1-0G-pBBawaVOf0GA2B5Vteq-I03Gy42V',
    'lgbtq': 'https://drive.google.com/uc?id=1-E4-ZUmJihYV9WJ5kwMUfyFzUKastebG',
    'liberal': 'https://drive.google.com/uc?id=1-Nbt8JWrgXBEkF3fceRkrPReGQefmM-7',
    'sports': 'https://drive.google.com/uc?id=1-Qb3USg32mQoWOi4nu1YnebxBfLJjSUN',
    'tech-nerd': 'https://drive.google.com/uc?id=1-7mnZi970TGltEe26oiTC4hRvV_DA-km'
}
urls_eachsub = {
    'academic-humanities': 'https://drive.google.com/uc?id=1-AIBNRQzu6NGCmaIkmdXhj_jNlYtn_Sz',
    'academic-stem': 'https://drive.google.com/uc?id=1tNwfOXHMYeHOaThRuChbXobC0Egob0hI',
    'anime': 'https://drive.google.com/uc?id=1--ZMk8J04IbnpwXLChBKFQtvsqYppjTt',
    'astrology': 'https://drive.google.com/uc?id=1025SE0dwAJ_7hviHpP7GzzGP8hdbc0zS',
    'conservative': 'https://drive.google.com/uc?id=1RGLJuqTtrAPQvWCBKa2zeJdDh5P7nqRB',
    'hippie-spiritual': 'https://drive.google.com/uc?id=1-6qmc-neLaiKagWXhJZlWn27Dnu6eY2Q',
    'kpop': 'https://drive.google.com/uc?id=1tNhAoNu8DSVPwJl1-ju5DTtnTDYTiX1y',
    'lgbtq': 'https://drive.google.com/uc?id=1-DWml_p85TQHPinu_W5_OXypiLot5KC4',
    'liberal': 'https://drive.google.com/uc?id=1-EkacP440CZW7CEXvQyh0sWTGfCax3oN',
    'sports': 'https://drive.google.com/uc?id=1-OMOSo7B-fru8tjN0foBpsiUc00N8Ukl',
    'tech-nerd': 'https://drive.google.com/uc?id=1S5AExvpdwGOIOQ4bdq3XfCH1-Y1-a4li'
}

class DownloadPkls:
    def __init__(self):
        pass
    
    def testfunc(self):
        return 'hello world!'

    def download_things(self, _url, _folder, _filename):
        output = os.path.join(path_cwd, 'static', _folder, _filename)
        #cwd is just the first pytorch-django folder...

        try:
            gdown.download(_url, str(output), quiet=False)
            return 'success!'
        except Exception as e:
            return e

    def download_all_models2(self):
        for i in range(0, len(subs)):
            url = urls_eachsub[subs[i]]
            folder = 'models'
            filename = 'nlpmodel3-' + subs[i] + '.pkl'
            if not os.path.exists(os.path.join(path_cwd, path_models, filename)):
                self.download_things(url, folder, filename)
    
    def download_all_models(self):
        for i in range(0, len(subs)):
            url = urls_eachsub[subs[i]]
            folder = 'models'
            filename = 'nlpmodel3-' + subs[i] + '.pth'
            if not os.path.exists(os.path.join(path_cwd, path_models, filename)):
                self.download_things(url, folder, filename)

    def download_toks200(self):
        url = 'https://drive.google.com/uc?id=1fx1HDjJ7O9Hryq6yu_AymzM5GtULcSWx'
        folder = 'toks200'
        filename = 'toks200-tweets.pkl'
        if not os.path.exists(os.path.join(path_cwd, path_toks200, filename)):
            self.download_things(url, folder, filename)

    def download_nums200(self):
        url = 'https://drive.google.com/uc?id=1IgIcw_CRJQgdTo-Nn4sxk_g5Vd3xqiob'
        folder = 'nums200'
        filename = 'nums200-eachsub.pkl'
        if not os.path.exists(os.path.join(path_cwd, path_nums200, filename)):
            self.download_things(url, folder, filename)
    
    def download_dls_c(self):
        url = 'https://drive.google.com/uc?id=1FITUFGf7BVi_-H8kITmiq5nxxGf8nSNs'
        folder = 'dataloaders'
        filename = 'dls-nlp-clas.pkl'
        if not os.path.exists(os.path.join(path_cwd, path_dls, filename)):
            self.download_things(url, folder, filename)

    def download_learn_c_pth(self):
        url = 'https://drive.google.com/uc?id=1mVSE6pLnItsEiNQ3gCLyxJcLA37bWzfW'
        folder = 'models'
        filename = 'nlpmodel3_clas.pth'
        if not os.path.exists(os.path.join(path_cwd, path_models, filename)):
            self.download_things(url, folder, filename)