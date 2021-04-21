from django.urls import path
from pose.views import PoseCheckView

urlpatterns = [
    path("", PoseCheckView.as_view()),
]