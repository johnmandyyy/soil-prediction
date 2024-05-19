from rest_framework.views import APIView
import shutil
from rest_framework.response import Response
import os
from app.builder.cnn_builder import CNN as Algorithm
from asgiref.sync import async_to_sync
import cv2
import base64
import numpy as np

CNN_Instance = Algorithm()


class Prediction(APIView):

    def __init__(self):
        pass

    def post(self, request, format=None):

        print(request.FILES["image"], type(request.FILES["image"]))

        uploaded_image = request.FILES["image"]

        # Read the uploaded image using OpenCV
        image_array = np.frombuffer(uploaded_image.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        np_frame, categ_name, probability = CNN_Instance.predict_image_v2(image)
        # [np_frame, class_names[highestProb], str(max(output[0]) * 100)]

        _, buffer = cv2.imencode(".jpg", np_frame)
        base64_image = base64.b64encode(buffer).decode()

        return Response(
            {
                "base64_image": base64_image,
                "category_list": categ_name,
                "probability": probability,
            },
            200,
        )


class CNN(APIView):

    def __init__(self):
        self.valid_actions = ["train", "predict"]

    def get(self, request, action=None, format=None):

        if action in self.valid_actions:

            if action == "train":
                CNN_Instance.train()

            return Response({"message": "Done Training", "action": action}, 200)
        else:
            return Response({"message": "Action is not valid.", "action": action}, 400)


class GetImages(APIView):

    def __init__(self):
        pass

    def get(self, request, obj_name=None, format=None):
        # Define the directory path
        print(obj_name)
        total_images = 0
        media_root = "media/train"
        entries = os.listdir(media_root)

        # Initialize an empty list to store folder names and their files
        folders = []

        # Iterate through each entry
        for entry in entries:
            # Construct the full path of the entry
            entry_path = os.path.join(media_root, entry)

            # Check if the entry is a directory
            if os.path.isdir(entry_path):
                # Initialize an empty list to store files within the directory
                files = []

                # List all files within the directory
                for file_name in os.listdir(entry_path):
                    # Construct the full path of the file
                    file_path = os.path.join(entry_path, file_name)

                    # Check if the entry is a file
                    if os.path.isfile(file_path):
                        # Append the file name to the list of files
                        files.append("/" + media_root + "/" + entry + "/" + file_name)
                        total_images = total_images + 1

                if obj_name == None:
                    folders.append({"folder_name": entry, "files": files})

                elif obj_name == entry:
                    folders.append({"folder_name": entry, "files": files})
        # Return the list of folders and their files
        return Response({"train_directory": folders, "total_images": total_images})
