from django.db import models
from django.contrib.postgres.fields import JSONField

# Create your models here.
class Target(models.Model):
    target_text = models.CharField(max_length=10)
    pub_date = models.DateTimeField('date Published')


class Therapeutic(models.Model):
    therapeutic_text = models.CharField(max_length=30)
    pub_date = models.DateTimeField('date Published')


class Indication(models.Model):
    indication_text = models.CharField(max_length=25)
    pub_date = models.DateTimeField('date Published')


class Model(models.Model):
    model_text = models.CharField(max_length=20)
    pub_date = models.DateTimeField('date Published')


class Compound(models.Model):
    compound_text = models.CharField(max_length=100)
    pub_date = models.DateTimeField('date Published')


class Similar(models.Model):
    name = models.CharField(max_length=100)
    user_id = models.IntegerField()
    type = models.CharField(max_length=100)
    image_id = models.CharField(max_length=100)
    similarity = models.FloatField()
    smile = models.CharField(max_length=10000)

class Prediction(models.Model):
    data = JSONField()
    user_id = models.IntegerField()
