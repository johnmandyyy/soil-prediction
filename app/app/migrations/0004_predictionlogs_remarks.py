# Generated by Django 4.2.2 on 2024-05-19 11:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_predictionlogs'),
    ]

    operations = [
        migrations.AddField(
            model_name='predictionlogs',
            name='remarks',
            field=models.TextField(null=True),
        ),
    ]
