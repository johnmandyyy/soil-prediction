from rest_framework import generics, response
from rest_framework.permissions import IsAdminUser
import json
from AesEverywhere import aes256
from django.conf import settings
from django.core.files.base import ContentFile
import base64

from rest_framework import generics
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.response import Response

from rest_framework.views import APIView
from rest_framework.response import Response as DR
from .authentication import BearerTokenAuthenticator as BTA

# ORM-able


class EncryptedResponse(response.Response):
    def __init__(self, data=None, status=None, *args, **kwargs):
        encrypted_data = aes256.encrypt(
            json.dumps(data), str(settings.SECRET_KEY)
        ).decode("utf-8")
        super().__init__({"data": encrypted_data}, status, *args, **kwargs)


class EncryptedAPIView(generics.GenericAPIView):
    lookup_field = "id"


class GET_POST(EncryptedAPIView, generics.ListCreateAPIView):
    queryset = None
    parser_classes = (JSONParser, MultiPartParser)
    serializer_class = None

    def get_response_data(self, message):
        return {"message": message}

    def list(self, request, *args, **kwargs):
        print("List executed.")
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return EncryptedResponse(serializer.data, status=200)

    def create(self, request, *args, **kwargs):
        encrypted_data = request.data.get("data")
        decrypted_data = ""
        try:
            decrypted_data = aes256.decrypt(
                encrypted_data, str(settings.SECRET_KEY)
            ).decode("utf-8")
            decrypted_json = json.loads(decrypted_data)
            image_base64 = decrypted_json.pop("image", None)
            if image_base64:
                image_data = base64.b64decode(image_base64)
                image_file = ContentFile(image_data, name="image.jpg")
                decrypted_json["image"] = image_file

        except Exception as e:
            error_response = self.get_response_data("Decryption failed.")
            error_response["error"] = str(e) + str(decrypted_data)
            return response.Response(error_response, status=400)

        serializer = self.get_serializer(data=decrypted_json)

        if serializer.is_valid():
            serializer.save()
            return response.Response(
                self.get_response_data("Object created successfully."), status=201
            )
        else:
            return response.Response(
                self.get_response_data(
                    "Serializer is is not valid." + str(decrypted_data)
                ),
                status=400,
            )


class GET_PUT_PATCH_DELETE(EncryptedAPIView, generics.RetrieveUpdateDestroyAPIView):

    queryset = None
    serializer_class = None

    def get_response_data(self, message):
        return {"message": message}

    def retrieve(self, request, *args, **kwargs):
        print("Retreived executed.")
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return EncryptedResponse(serializer.data, status=200)

    def update(self, request, *args, **kwargs):
        instance = self.get_object()

        encrypted_data = request.data.get("data")
        decrypted_data = ""
        try:
            decrypted_data = aes256.decrypt(
                encrypted_data, str(settings.SECRET_KEY)
            ).decode("utf-8")
            decrypted_json = json.loads(decrypted_data)
        except Exception as e:
            error_response = self.get_response_data("Decryption failed.")
            error_response["error"] = str(e) + str(decrypted_data)
            return response.Response(error_response, status=400)

        serializer = self.get_serializer(instance, data=decrypted_json)

        if serializer.is_valid():
            serializer.save()
            return response.Response(
                self.get_response_data("Object updated successfully."), status=200
            )
        else:
            return response.Response(
                self.get_response_data(
                    "Serializer is not valid." + str(decrypted_data)
                ),
                status=400,
            )

    def destroy(self, request, *args, **kwargs):

        if BTA.authenticate_with_bearer_token(request).is_superuser == True:

            instance = self.get_object()
            instance.delete()

            return response.Response(
                self.get_response_data("Object deleted successfully."), status=204
            )

        else:

            return DR({"message": "Forbidden permission/s"}, 403)


class GET(EncryptedAPIView, generics.ListAPIView):
    queryset = None
    parser_classes = (JSONParser, MultiPartParser)
    serializer_class = None

    def get_response_data(self, message):
        return {"message": message}

    def list(self, request, *args, **kwargs):
        print("List executed.")
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return EncryptedResponse(serializer.data, status=200)


# RAW
class GET_RAW(APIView):
    success_response = {}

    def __init__(self):
        pass

    def get(self, request, format=None):
        try:
            return EncryptedResponse(self.success_response, 200)
        except Exception as e:
            return DR({"message": str(e)}, 400)
