# review_app/models.py
from django.db import models

class Review(models.Model):
    reviewText = models.TextField()
    vote = models.IntegerField()
    # Add other fields as needed
    
    def __str__(self):
        return self.reviewText

