# Generated by Django 3.1.7 on 2021-03-12 16:20

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('drugs', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Compounds',
            new_name='Compound',
        ),
        migrations.RenameModel(
            old_name='Indications',
            new_name='Indication',
        ),
        migrations.RenameModel(
            old_name='Models',
            new_name='Model',
        ),
        migrations.RenameModel(
            old_name='Targets',
            new_name='Target',
        ),
        migrations.RenameModel(
            old_name='Images',
            new_name='Image',
        ),
    ]