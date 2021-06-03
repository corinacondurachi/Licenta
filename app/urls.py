
from django.urls import path
from . import views

from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *

urlpatterns = [
    path('', views.home, name = 'app-home'),
    path('image_upload', picture_image_view, name = 'image_upload'),
    path('success', success, name = 'success'),
    path('get_colorized_images', get_colorized_images),
]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)