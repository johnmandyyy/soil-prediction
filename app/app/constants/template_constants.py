from app.builder.template_builder import Builder

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

PREDICTION = (
    Builder()
    .addPage("app/predict.html")
    .addTitle("Predict Soil Image")
)

REPORTS = (
    Builder()
    .addPage("app/reports.html")
    .addTitle("Reports")
)

REPORTS.build()
PREDICTION.build()
LOGIN.build()
INDEX.build()
DATASETS.build()

