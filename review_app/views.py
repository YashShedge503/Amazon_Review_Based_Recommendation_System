#from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from .models import Review
from .amazon_review_model import load_data, preprocess_data, train_model, predict
import json

# Load and train model once when the server starts (for efficiency)
model = None
X_test = None
y_test = None

def initialize_model():
    global model, X_test, y_test
    data = load_data('path_to_your_data.json')  # Adjust file path
    X_tfidf, y = preprocess_data(data)
    model, X_test, y_test = train_model(X_tfidf, y)

initialize_model()  # Call it to initialize at the start of the server

def predict_review(request):
    # Example: Send POST request with review text
    if request.method == 'POST':
        review_text = request.POST.get('reviewText', None)
        if review_text:
            # Convert the review text to a DataFrame, process it, and make prediction
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            review_tfidf = tfidf_vectorizer.fit_transform([review_text])
            prediction = predict(model, review_tfidf)
            return JsonResponse({'predicted_vote': prediction[0]})
        else:
            return JsonResponse({'error': 'No review text provided'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
