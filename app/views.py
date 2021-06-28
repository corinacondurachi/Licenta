from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import *
from .models import *
#import models

# Create your views here.

def home(request):
	return render(request, 'app/home.html')

def picture_image_view(request):
  
    if request.method == 'POST':
        form = PictureForm(request.POST, request.FILES)
  
        if form.is_valid():
            form.save()
            #return redirect('success')
            return render(request, 'app/picture_image_form.html', {'form' : form})
    else:
        form = PictureForm()
    return render(request, 'app/picture_image_form.html', {'form' : form})
  

def get_colorized_images(request):

    generator = build_second_generator(chanels_input=1, chanels_output=2, size=256)
    generator.load_state_dict(torch.load('res18-unet', map_location=torch.device('cpu')))

    model = MainModel(net_G=generator)
    model.load_state_dict(torch.load('final_model_80ep.pt', map_location=torch.device('cpu')))

    val_paths = []
    path = 'media/images/'
    for file in os.listdir(path):
        val_paths.append (path + file)
    if path + '.DS_Store' in val_paths:
        val_paths.remove(path + '.DS_Store')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    see_results(model,val_dl, val_paths,'second_')
    print("finised")
    return render(request, 'app/picture_image_form.html')


def get_colorized_images_first(request):

    
    model = MainModel()
    model.load_state_dict(torch.load('first_model', map_location=torch.device('cpu')))

    val_paths = []
    path = 'media/images/'
    for file in os.listdir(path):
        val_paths.append (path + file)
    if path + '.DS_Store' in val_paths:
        val_paths.remove(path + '.DS_Store')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    see_results(model, val_dl, val_paths,'first_')
    print("finised")
    empty_folder('media/images/')
    return render(request, 'app/picture_image_form.html')


def success(request):
    return HttpResponse('successfully uploaded')