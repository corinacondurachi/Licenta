from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import *

# Create your views here.

def home(request):
	return render(request, 'app/home.html')

def picture_image_view(request):
  
    if request.method == 'POST':
        form = PictureForm(request.POST, request.FILES)
  
        if form.is_valid():
            form.save()
            return redirect('success')
    else:
        form = PictureForm()
    return render(request, 'app/picture_image_form.html', {'form' : form})
  
  
def success(request):
    return HttpResponse('successfully uploaded')