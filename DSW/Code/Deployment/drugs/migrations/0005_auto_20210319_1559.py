# Generated by Django 3.1.7 on 2021-03-19 20:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('drugs', '0004_therapeutic_area'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Therapeutic_area',
            new_name='Therapeutic',
        ),
        migrations.RenameField(
            model_name='therapeutic',
            old_name='target_text',
            new_name='therapeutic_text',
        ),
    ]
