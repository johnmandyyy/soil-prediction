# Generated by Django 4.2.5 on 2024-05-18 16:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='categories',
            name='folder_name',
            field=models.TextField(null=True),
        ),
    ]
