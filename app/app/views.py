"""
Definition of views.
"""

from django.shortcuts import render, redirect
from django.http import HttpRequest
import app.constants.template_constants as Templates
from django.contrib.auth import logout, login, authenticate


class TemplateView:
    """A template level views."""

    def __init__(self):
        pass


    def reports(self, request):
        """Renders the reports page."""

        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return redirect("login")

        return Templates.REPORTS.render_page(request)
    
    def home(self, request):
        """Renders the home page."""

        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return redirect("login")

        return Templates.INDEX.render_page(request)

    def datasets(self, request):
        """Renders the home page."""

        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return redirect("login")

        return Templates.DATASETS.render_page(request)

    def predict(self, request):
        """Renders the home page."""

        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return redirect("login")

        return Templates.PREDICTION.render_page(request)

    def login(self, request):
        """Renders the home page."""

        assert isinstance(request, HttpRequest)

        if request.user.is_authenticated == False:
            return Templates.LOGIN.render_page(request)

        return redirect("home")
    
    def authenticate_user(self, request):
        try:

            if request.method == 'POST':
                
                username = request.POST.get('username')
                password = request.POST.get('password')
                user = authenticate(request, username=username, password=password)

                if user is not None:
                    login(request, user)
                    return redirect('home')
                
        except Exception as e:
            pass

        return redirect('login')
            

    def user_logout(self, request):
        logout(request)
        return redirect("login")
