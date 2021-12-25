from django.db import models

class Topic(models.Model):
    topic_text = models.CharField(max_length=20)
    def __str__(self):
        return self.topic_text

class Tweet(models.Model):
    tweet = models.ForeignKey(Topic, on_delete=models.CASCADE)
    tweet_text = models.CharField(max_length=250)
    def __str__(self):
        return self.tweet_text