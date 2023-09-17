from django.urls import path
from . import views

urlpatterns = [
    path('decisiontree/', views.decision_tree),
    path('decisiontree/getResults/', views.get_results),
    path('predictor/', views.predictor),
    path('predictor/getPrediction/', views.get_prediction)
]