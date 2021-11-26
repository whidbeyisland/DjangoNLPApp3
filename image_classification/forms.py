from django import forms


class TextEntryForm(forms.Form):
    prompt = forms.CharField()
