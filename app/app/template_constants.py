from datetime import datetime
from django.shortcuts import render
from app.template_builder import Builder


LOGIN = (
    Builder()
    .addPage("app/login.html")
    .addTitle("Login Page")
)

INDEX = (
    Builder()
    .addPage("app/index.html")
    .addTitle("Dashboard Page")
)

DATASETS = (
    Builder()
    .addPage("app/datasets.html")
    .addTitle("Manage Datasets")
)

LOGIN.build()
INDEX.build()
DATASETS.build()

