from django.contrib import admin
from drugs.models import Target, Indication, Model, Compound, Therapeutic
admin.site.register(Target)
admin.site.register(Therapeutic)
admin.site.register(Indication)
admin.site.register(Model)
admin.site.register(Compound)
# Register your models here.
