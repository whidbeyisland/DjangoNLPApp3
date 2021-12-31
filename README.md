# Parrot: Tweet Generation App

Parrot is an experimental app for generating fake tweets in the style of a chosen Twitter user, about a chosen topic.

To create viable tweets, Parrot leverages several modern machine learning and natural language processing technologies, including the FastAI framework for PyTorch, purpose-built generative and classificatory language models, NLTK's part-of-speech tagger, and the WordNet database of lexical similarity.

The Python business logic is hosted in a heavily modified version of "Pytorch-Django" by Stefan Schneider, as a way of quickly getting a Django front-end up and running. All models used were developed externally in Colab Notebooks.

<b>NOTE:</b> Parrot has only been tested on Windows 10, and compatibility with other operating systems cannot be guaranteed.

<img src="docs/elon_musk_fine_art.PNG" width="50%" />
<img src="docs/jimmy_wales_spiders.PNG" width="50%" />

## Setup

```
pip install -r requirements
```

### Development

```
python manage.py runserver
```
The app is running on http://localhost:8000/
