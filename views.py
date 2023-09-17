from django.shortcuts import render
from django.http import HttpResponse
from . import decisiontree

# Create your views here.

def decision_tree(request):
    return render(request, 'evaluator_dashboard.html')

def get_results(request):
    file = request.FILES['file']
    k = int(request.POST['cv'])
    depth = int(request.POST['depth'])
    prune = int(request.POST['min'])
    impurity = request.POST['impurity']
    accuracy = decisiontree.calculate(file, k, depth, prune, impurity)
    return render(request, 'results.html', {'accuracy': accuracy})

def predictor(request):
    return render(request, 'predictor_dashboard.html')

def get_prediction(request):
    age = int(request.POST['age'])
    gender = request.POST['gender']
    polyuria = request.POST['polyuria']
    polydipsia = request.POST['polydipsia']
    sudden_weight_loss = request.POST['sudden_weight_loss']
    weakness = request.POST['weakness']
    polyphagia = request.POST['polyphagia']
    genital_thrush = request.POST['genital_thrush']
    visual_blurring = request.POST['visual_blurring']
    itching = request.POST['itching']
    irritability = request.POST['irritability']
    delayed_healing = request.POST['delayed_healing']
    partial_paresis = request.POST['partial_paresis']
    muscle_stiffness = request.POST['muscle_stiffness']
    alopecia = request.POST['alopecia']
    obesity =request.POST['obesity']
    prediction = decisiontree.predict_diabetes(age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, genital_thrush, visual_blurring, itching, irritability, delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity)
    return render(request, 'prediction.html', {'prediction':prediction})
