from rest_framework import serializers
from django.apps import apps
from app.models import models

class ModelHelpers:
    """ A predefined for model helpers. """
    def __init__(self):
        self.predefined_models = None
        self.__initialize()

    def get_model_instance(self, model_name):
        try:
            print(model_name)
            return apps.get_model("app", model_name)
        except Exception as e:
            return None

    def __initialize(self):

        # Get a list of all installed models
        installed_models = apps.get_models()
        self.predefined_models = []

        for model in installed_models:
            # Get the name of the model and append it to the list
            if model.__name__ not in [
                "LogEntry",
                "Permission",
                "Group",
                "User",
                "ContentType",
                "Session",
            ]:
                self.predefined_models.append(model.__name__)

class SerializerHelpers:
    """ A predefined for serializer helpers. """

    def __init__(self):
        pass

    def create_serializer(self, model_name: models, app_name: str) -> serializers:
        """
        Create an automatic serializer for your API.
        """
        django_model = apps.get_model(app_label=str(app_name), model_name=model_name)
        print(django_model, "Autoserializer was built.")

        class AutoSerializer(serializers.ModelSerializer):

            class Meta:
                model = django_model
                fields = "__all__"

        return AutoSerializer
