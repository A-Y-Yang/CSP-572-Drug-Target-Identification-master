from django.contrib.auth.forms import AuthenticationForm
from django import forms


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super(LoginForm, self).__init__(*args, **kwargs)

    username = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'username'}), label='')
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'password'}), label='')
