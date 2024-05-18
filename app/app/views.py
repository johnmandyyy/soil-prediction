"""
Definition of views.
"""
from django.shortcuts import render, redirect
from datetime import datetime
from django.http import HttpRequest
import app.template_constants as Templates

class TemplateView:

    def __init__(self):
        pass

    def home(self, request):
        """Renders the home page."""

        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return Templates.LOGIN.render_page(request) 
        
        return Templates.INDEX.render_page(request)
    
    def datasets(self, request):
        """Renders the home page."""

        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return Templates.LOGIN.render_page(request) 
        
        return Templates.DATASETS.render_page(request) 
        
        
