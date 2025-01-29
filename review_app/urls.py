# review_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_review, name='predict_review'),
]
