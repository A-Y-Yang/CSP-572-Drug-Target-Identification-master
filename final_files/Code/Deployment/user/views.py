from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from user.registerationForm import RegistrationForm
from django.contrib import messages
from user.forms import LoginForm
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from drugs import views as drug_view
from django.conf import settings
from django.template.loader import render_to_string, get_template
from django.core.mail import EmailMessage, send_mail
from django.contrib.auth.tokens import PasswordResetTokenGenerator, default_token_generator
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
import requests
import after_response
import os
import mimetypes
from django.http import HttpResponse
from django import forms
from django.urls import reverse




def password_change(request):
    return render(request, 'drugs/password_reset_form.html')


def login_page(request):
    form = LoginForm()
    flag = render(request, 'drugs/login.html', {'form': form})
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        userr = authenticate(request, username=username, password=password)
        if userr is not None:
            if userr.is_active:
                login(request, userr)
                usser = User.objects.get(username=username)
                flag = redirect('dashboard')
            else:
                messages.error(request, f'You are not active user. Please contract administrator.')
                return render(request, 'drugs/login.html', {'form': form})
        else:
            messages.error(request, f'Invalid username or password.')
            return render(request, 'drugs/login.html', {'form': form})
    else:
        return flag
    return flag


def signup(request):
    if request.method == 'POST':
        username = request.POST['reg_first_name']
        email = request.POST['reg_email']
        password = request.POST['reg_password']
        confirm_password = request.POST['reg_confirm_password']
        if confirm_password == password:
            user = User.objects.create_user(username=email, email=email, first_name=username, password=password)
            user.save()
            messages.success(request, f'Account created for {username}!')
            return redirect('/login')
        else:
            messages.error(request, f'Password did not match.')
            return redirect('/signup')
    else:
        form = RegistrationForm()
    return render(request, 'drugs/signup.html', {'form': form})


def log_out(request):
    logout(request)
    messages.success(request, f'Successfully logged out.')
    # after_response.drug_view.ml_model()
    return redirect('/login')


def password_reset(request):
    if request.method == 'POST':
        email = request.POST.get('emaiil')
        domain = request.headers['Host']
        if User.objects.filter(email=email).exists():
            usr = User.objects.get(email=email)
            subject = "Password Reset Requested"
            email_template_name = "drugs/password_reset_email.html"
            c = {
                "email": usr.email,
                'domain': domain,
                'site_name': 'Interface',
                "uid": urlsafe_base64_encode(force_bytes(usr.pk)),
                "user": usr,
                'token': default_token_generator.make_token(usr),
                'protocol': 'http',
            }
            stri = ''+c['protocol']+'://'+c['domain']+'/password-reset-confirm/'+c['uid']+'/'+c['token']+'/'
            print(stri)
            emmail = render_to_string(email_template_name, c)
            from_email = settings.EMAIL_HOST_USER
            send_mail(subject, emmail, from_email, [usr.email], fail_silently=False)
            # html_template = 'drugs/password_reset_email.html'
            # html_message = render_to_string(html_template)
            # mail = EmailMessage(subject, html_message, from_email, recipient_list)
            # mail.send()
            messages.success(request, 'We have emailed you instructions for setting your password. '
                                      'You should receive the email shortly!')
            return render(request, 'drugs/password_reset_form.html')
        else:
            messages.error(request, "User does not exist in our database.")
            print("User does not exist!")
            return render(request, 'drugs/password_reset_form.html')
    else:
        return render(request, "drugs/password_reset_form.html")


def passreconfirm():
    print("hola")
    print("QWERTY: ", requests.get_raw_uri())
    if requests.method == 'POST':
        pass1 = requests.POST.get('passw1')
        pass2 = requests.POST.get('passw2')
        if pass1 == pass2:
            if len(pass1) > 8:
                print(requests.GET.get('token'))
                login(requests)



@login_required(login_url='login')
def dashboard(request):
    drug_view.dashboard(request)
