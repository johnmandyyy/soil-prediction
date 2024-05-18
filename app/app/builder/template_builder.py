from django.shortcuts import render
from datetime import datetime

class TemplateBuilder:

    """ A template builder class. """
    def __init__(self):

        self.__Page = None
        self.__Title = None

    def setPage(self, page: str) -> None:
        """A method used to set page location"""
        self.__Page = page

    def setTitle(self, title: str) -> None:
        """A method used to set title"""
        self.__Title = title

    def getProps(self):
        return {"page": self.__Page, "title": self.__Title}


class Builder:

    def __init__(self):
        self.instance = TemplateBuilder()
        self.Page = None
        self.Title = None

    def addPage(self, ingridients) -> TemplateBuilder:
        self.instance.setPage(ingridients)
        return self

    def addTitle(self, ingridients) -> TemplateBuilder:
        self.instance.setTitle(ingridients)
        return self

    def build(self) -> TemplateBuilder:
        """A method used to build the object."""
        self.Page = self.instance.getProps()["page"]
        self.Title = {
            "title": self.instance.getProps()["title"],
            "date": str(datetime.now()),
        }
        return self.instance

    def render_page(self, request):
        """ A method to render when there is an error in the page. """
        try:
            print(request, self.Page, self.Title)
            return render(request, self.Page, self.Title)
        except Exception as e:
            # Render an Error Page
            print(e)
            return render(
                request,
                "app/constants/error.html",
                {
                    "title": "Maintenance Page",
                    "date": str(datetime.now()),
                    "message": str(e),
                },
            )
