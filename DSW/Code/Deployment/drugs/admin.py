from django.contrib import admin
from .models import Target, Indication, Model, Compound, Therapeutic, Similar, Prediction
admin.site.register(Target)
admin.site.register(Therapeutic)
admin.site.register(Indication)
admin.site.register(Model)
admin.site.register(Compound)
admin.site.register(Similar)
admin.site.register(Prediction)
# Register your models here.
