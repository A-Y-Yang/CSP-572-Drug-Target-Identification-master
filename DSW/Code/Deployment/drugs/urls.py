from django.urls import path
from drugs import views as dview
from drugs import results, about, faq, manual

from user import views as uview
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', dview.dashboard, name='dashboard'),
    path('login/', uview.login_page, name='login'),
    path('logout/', uview.log_out, name='logout'),
    path('signup/', uview.signup, name='signup'),
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='drugs/password_reset_form.html'), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='drugs/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='drugs/password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='drugs/password_reset_complete.html'), name='password_reset_complete'),
    path('results/', results.index, name="results"),
    path('about/', about.index, name="about"),
    path('faq/', faq.index, name='faq'),
    path('manual/', manual.index, name='manual')
]