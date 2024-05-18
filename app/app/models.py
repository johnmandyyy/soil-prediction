"""
Definition of models.
"""

from django.db import models

# Create your models here.

class Categories(models.Model):
    category_name = models.TextField()

class Reports(models.Model):
    accuracy = models.TextField()
    precision = models.TextField()
    recall = models.TextField()
    f1_score = models.TextField()


