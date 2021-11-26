from django import forms


class ImageUploadForm(forms.Form):
    prompt = forms.CharField()
    image = forms.ImageField()
