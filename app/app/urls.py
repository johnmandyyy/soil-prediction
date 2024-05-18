"""
Definition of urls for app.
"""

from django.urls import path, include
from django.contrib import admin
from app import views
from .api import *
import app.constants.url_constants as URLConstants
from app.constants import app_constants

MainView = views.TemplateView()

list_create_patterns = URLConstants.GenericAPI.list_create_patterns
get_update_destroy_patterns = URLConstants.GenericAPI.retrieve_update_delete_patterns

api_patterns = [
    path("api/", include((list_create_patterns, app_constants.APP_NAME))),
    path("api/", include((get_update_destroy_patterns, app_constants.APP_NAME))),
]

template_patterns = [
    path("", MainView.home, name="home"),
    path("predict/", MainView.predict, name="predict"),
    path("datasets/", MainView.datasets, name="dataset"),
    path("login/", MainView.login, name="login"),
    path("logout/", MainView.user_logout, name="logout"),
    path("admin/", admin.site.urls),
]

urlpatterns = template_patterns + api_patterns
