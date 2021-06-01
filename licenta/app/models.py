from django.db import models

# Create your models here.

class Picture(models.Model):
    name = models.CharField(max_length=50)
    picture_Main_Img = models.ImageField(upload_to='images/')