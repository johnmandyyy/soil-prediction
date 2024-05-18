"""
Definition of urls for app.
"""

from datetime import datetime
from django.urls import path
from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from app import forms, views


MainView = views.TemplateView()


urlpatterns = [
    path('', MainView.home, name='home'),
    path('datasets/', MainView.datasets, name='dataset'),
    path('admin/', admin.site.urls),
]
