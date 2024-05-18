from app.helpers.helpers import ModelHelpers
from django.urls import path
from app.builder.api_builder import APIBuilder
from app.constants import app_constants

class UrlPatternBuilder:

    def __init__(self):

        self.list_create_patterns = []
        self.retrieve_update_delete_patterns = []

    def build(self, name=app_constants.APP_NAME):

        django_models = ModelHelpers()

        for each_models in django_models.predefined_models:

            model_url_name = each_models.lower()

            list_create_url = "list-create/" + str(model_url_name) + "/"
            get_update_destroy_url = (
                "get-update-destroy"
                + "/"
                + str(model_url_name)
                + "/"
                + "<int:pk>"
                + "/"
            )

            built_object_instance = APIBuilder(
                each_models, name, ModelHelpers().get_model_instance(each_models)
            )
            built_object_instance.build()

            self.list_create_patterns.append(
                path(
                    list_create_url,
                    built_object_instance.list_create.as_view(),
                    name="list-create" + "-" + model_url_name,
                )
            )

            self.retrieve_update_delete_patterns.append(
                path(
                    get_update_destroy_url,
                    built_object_instance.get_update_destroy.as_view(),
                    name="get-update-delete-categories" + "-" + model_url_name,
                )
            )
