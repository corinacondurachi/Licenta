
# forms.py
from django import forms
from .models import *
  
class PictureForm(forms.ModelForm):
  
    class Meta:
        model = Picture
        fields = ['picture_Main_Img']