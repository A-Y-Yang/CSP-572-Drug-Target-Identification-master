# Generated by Django 3.1.7 on 2021-03-19 20:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('drugs', '0003_auto_20210319_1545'),
    ]

    operations = [
        migrations.CreateModel(
            name='Therapeutic_area',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('target_text', models.CharField(max_length=30)),
                ('pub_date', models.DateTimeField(verbose_name='date Published')),
            ],
        ),
    ]
