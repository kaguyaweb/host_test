from django.urls import path

from . import views

#urlpatterns = [
#    # ex: /polls/
#    path('', views.index, name='index'),
#    # ex: /polls/5/
#    path('specifics/<int:question_id>/', views.detail, name='detail'),
#    # ex: /polls/5/results/
#    path('<int:question_id>/results/', views.results, name='results'),
#    # ex: /polls/5/vote/
#    path('<int:question_id>/vote/', views.vote, name='vote'),
#]

#app_name = 'polls'
#urlpatterns = [
#    path('', views.IndexView.as_view(), name='index'),
#    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
#    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
#    path('<int:question_id>/vote/', views.vote, name='vote'),
#    # added the word 'specifics'
#    #path('specifics/<int:question_id>/', views.detail, name='detail'),
#]

app_name = 'polls'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'), #はじめのページ
    #path('topic/', views.DetailView.as_view(), name='detail'),
    #path('topic/analysis', views.ResultsView.as_view(), name='detail'),
    path('topic/', views.detail, name='detail'),
    path('topic_no_analysis/', views.detail2, name='detail2'),
    path('topic_3/', views.detail3, name='detail3'),
    path('topic/analysis/', views.analysis_tweet, name='results'),
    path('topic_3/analysis_no_others/', views.analysis_tweet2, name='results2'),
    #path('<int:question_id>/vote/', views.vote, name='vote'),
    # added the word 'specifics'
    #path('specifics/<int:question_id>/', views.detail, name='detail'),
]

