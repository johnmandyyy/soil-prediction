# Generated by Django 4.2.2 on 2024-05-19 11:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_categories_folder_name'),
    ]

    operations = [
        migrations.CreateModel(
            name='PredictionLogs',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('correct_answer', models.TextField(null=True)),
                ('forecasted', models.TextField(null=True)),
            ],
        ),
    ]
