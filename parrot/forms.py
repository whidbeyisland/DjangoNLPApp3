from django import forms


class TextEntryForm(forms.Form):
    username = forms.CharField()
    prompt = forms.CharField()
