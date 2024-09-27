from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.hello,name='hello'),
    path('tx',views.tex,name='tex'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('index/', views.index, name='index'),
]