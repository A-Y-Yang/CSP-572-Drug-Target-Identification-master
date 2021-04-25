from django.contrib.auth.forms import UserCreationForm
from django import forms


class RegistrationForm(UserCreationForm):
    def __init__(self, *args, **kwargs):
        super(RegistrationForm, self).__init__(*args, **kwargs)

    reg_first_name = forms.CharField(widget=forms.TextInput(attrs={'class': 'reg_first_name'}), label='')
    reg_email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'reg_email'}), label='')
    reg_password =  forms.CharField(widget=forms.PasswordInput(attrs={'class': 'reg_password'}), label='')
    reg_confirm_password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'reg_confirm_password'}), label='')