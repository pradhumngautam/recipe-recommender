from django.http import HttpResponse
from django.shortcuts import render
import joblib

def home(request):
    return render(request,"home.html")

def result(request):

    cls = joblib.load ('tfidf_vectorizer.pkl')
    
    lis = []
    lis.append(request.GET[IN])

    ans = cls.predict([lis])

    return render (request, "result.html",{'ans':ans})