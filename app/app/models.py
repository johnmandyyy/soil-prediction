"""
Definition of models.
"""

from django.db import models
import os
import shutil
# Create your models here.


class Categories(models.Model):

    category_name = models.TextField()
    folder_name = models.TextField(null = True)

    def save(self, *args, **kwargs):
        # Ensure category_name is in lowercase and create folder_name
        self.folder_name = self.category_name.lower().replace(' ', '')
        super(Categories, self).save(*args, **kwargs)
        # Create the directory inside media/train
        media_root = 'media/train'
        folder_path = os.path.join(media_root, self.folder_name)
        os.makedirs(folder_path)

    def delete(self, *args, **kwargs):
        super(Categories, self).delete(*args, **kwargs)
        media_root = 'media/train'
        folder_path = os.path.join(media_root, self.folder_name)
        shutil.rmtree(folder_path)

class PredictionLogs(models.Model):
    correct_answer = models.TextField(null = True)
    forecasted = models.TextField(null = True)
    remarks = models.TextField(null = True)

class Reports(models.Model):
    accuracy = models.TextField()
    precision = models.TextField()
    recall = models.TextField()
    f1_score = models.TextField()


