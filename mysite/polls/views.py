from django.shortcuts import render
from django.views import generic
import io
import re
import json
from requests_oauthlib import OAuth1Session
from janome.tokenizer import Tokenizer
import codecs
import numpy as np
import matplotlib.pyplot as plt
import random

def get_tweet(query):
    CONSUMER_KEY = 'afyFGi1n0yw5hfoCNZf1BVB9M'
    CONSUMER_SECRET = '4j2M8l4kYQ1Dr5Jwch3saNgRherc1FZI6lvGxNIE6qLkw5TeFl'
    ACCESS_TOKEN = '1423191244440674304-jl15GqAS4MFxiLOAbFldofkcXFyhbp'
    ACCESS_TOKEN_SECRET = 'vWoJIXNorBimElWPh88ymvEHHS3ZnxQ7XeVfCxZwBhEeS'
    twitter = OAuth1Session(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    # Twitter Endpoint(検索結果を取得する)
    url = 'https://api.twitter.com/1.1/search/tweets.json?tweet_mode=extended'

    # (min_faves:100 OR min_retweets:100 OR min_replies:10)

    keyword = query + ' -filter:replies -filter:images lang:ja'
    params ={
         'count' : 100,      # 取得するtweet数
         'exclude': 'retweets',  #RTを除外
         'q'     : keyword  # 検索キーワード
        }

    req = twitter.get(url, params = params)
    # 文章リスト
    texts = []

    if req.status_code == 200:
        res = json.loads(req.text)
        for line in res['statuses']:
            # 後ろのURL削除
            text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", line['full_text'])
            text = re.sub('@.+:\s', "", line['full_text'])
            texts.append(text)

    return texts

def random_list(texts):
    return random.sample(texts, 30)

def detail(request):
    query = request.POST['topic']
    texts = get_tweet(query)
    copy_texts = texts
    copy_texts = ','.join(copy_texts)

    context = {
        'topic' : query,
        'texts' : texts,
        'copy_texts' : copy_texts
    }
    return render(request, 'polls/detail.html', context)

def detail2(request):
    query = request.POST['topic']
    texts = get_tweet(query)
    copy_texts = texts
    copy_texts = ','.join(copy_texts)

    context = {
        'topic' : query,
        'texts' : texts,
    }
    return render(request, 'polls/detail2.html', context)

def detail3(request):
    query = request.POST['topic']
    texts = get_tweet(query)
    copy_texts = texts
    copy_texts = ','.join(copy_texts)

    context = {
        'topic' : query,
        'texts' : texts,
        'copy_texts' : copy_texts
    }
    return render(request, 'polls/detail3.html', context)

# JIWC-A_2018.csvファイルから、単語をキー、極性値を値とする辞書を得る
def load_pn_dict():
    dic = {}

    with codecs.open('/home/kagcom/host_test/mysite/polls/JIWC-A_2018-2019.csv', 'r', 'shift-jis') as f:
        lines = f.readlines()

        TrueFlag = True
        for line in lines:
            # 各行は"良い:よい:形容詞:0.999995"
            # 1行目は値じゃないのでスキップ
            if(TrueFlag):
                TrueFlag = False
                continue
            columns = line.split(',')
            #Sadness,Anxiety,Anger,Disgust,Trust,Surprise,Joy
            #悲しみ,不安,怒り,嫌悪感,信頼,驚き,喜び
            dic[columns[0]] = [float(columns[1]),float(columns[2]),float(columns[3]),float(columns[4]),float(columns[5]),float(columns[6]),float(columns[7])]

    return dic


# トークンリストから極性値リストを得る
def get_pn_scores(tokens, pn_dic):
    scores = []

    for surface in [t.surface for t in tokens if t.part_of_speech.split(',')[0] in ['動詞','名詞', '形容詞', '副詞', '形容動詞']]:
        if surface in pn_dic:
            scores.append(pn_dic[surface])

    return scores

def python_list_add(lst1, lst2):
    #np.set_printoptions(precision=4)
    work = np.array(lst1) + np.array(lst2)
    return work.tolist()


def get_pn_scores_sum(tokens, pn_dic):
    #初期化することで、もし感情辞書に載っていない単語しかない文章が出てきてもエラー回避できる
    scores = [0, 0, 0, 0, 0, 0, 0]
    count = 0.0001

    Firstflag = True
    for surface in [t.surface for t in tokens if t.part_of_speech.split(',')[0] in ['動詞','名詞', '形容詞', '副詞', '形容動詞']]:
        if Firstflag:
            if surface in pn_dic:
                Firstflag = False
                scores = pn_dic[surface]
                count += 1
                #print(word_count)
        else:
            if surface in pn_dic:
                scores = python_list_add(scores, pn_dic[surface])
                count += 1
                #print(word_count)

    return scores, count

def get_word_count(tokens, pn_dic):
    count = 0
    for surface in [t.surface for t in tokens if t.part_of_speech.split(',')[0] in ['動詞','名詞', '形容詞', '副詞', '形容動詞']]:
        if surface in pn_dic:
            count += 1

    return count

def find_min(text_score):
    index_min = []
    min = 0
    for i in range(len(text_score)):
        if text_score[min] > text_score[i]:
            min = i
    index_min.append(min)

    for i in range(len(text_score)):
        if text_score[min] == text_score[i] and min != i:
            index_min.append(i)

    return index_min

def find_not_max(text_score):
    index_not_max = []
    max = 0
    for i in range(len(text_score)):
        if text_score[max] < text_score[i]:
            max = i

    for i in range(len(text_score)):
        if i is not max or text_score[max] == 0.0:
            index_not_max.append(i)

    return index_not_max


def text_find(texts, min):
    element = sorted(texts, key=lambda e: e.pn_scores[min]/e.word_count, reverse=True)
    i = random.randint(0,4)
    return io.StringIO(element[i].text).read()

def no_word(texts):
    count = 0
    count_index = []
    for i in range(len(texts)):
        if texts[i].word_count == 0.0001:
            count += 1
            count_index.append(i)

    return count, count_index

def show_graph(element):
    ## labels = ["悲しみ", "不安", "怒り", "嫌悪感", "信頼", "驚き", "喜び"]
    labels = ["Sadness", "Anxiety", "Anger", "Disgust", "Trust", "Surprise", "Joy"]

    values = np.array([element.pn_scores[0]/element.word_count, element.pn_scores[1]/element.word_count, element.pn_scores[2]/element.word_count,
                       element.pn_scores[3]/element.word_count, element.pn_scores[4]/element.word_count, element.pn_scores[5]/element.word_count,
                       element.pn_scores[6]/element.word_count])
    # 多角形を閉じるためにデータの最後に最初の値を追加する。
    radar_values = np.concatenate([values, [values[0]]])
    # プロットする角度を生成する。
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)

    fig = plt.figure(facecolor="w")
    # 極座標でaxを作成。
    ax = fig.add_subplot(1, 1, 1, polar=True)
    # レーダーチャートの線を引く
    ax.plot(angles, radar_values)
    #　レーダーチャートの内側を塗りつぶす
    ax.fill(angles, radar_values, alpha=0.2)
    # 項目ラベルの表示
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)

    ax.set_title("radar chart", pad=20)
    ## print('Sentence: {}'.format(io.StringIO(element.text).read()))
    ## plt.show()
    plt.savefig('/home/kagcom/host_test/mysite/static/polls/out.png')

def analysis_tweet(request):
    class CorpusElement:
        def __init__(self, text='', tokens=[], pn_scores=[], word_count=float()):
            self.text = text # テキスト本文
            self.tokens = tokens # 構文木解析されたトークンのリスト
            self.pn_scores = pn_scores # 感情極性値(後述)
            self.word_count = word_count

    tweet_text = request.POST['tweet_text']
    text_list = request.POST.get('text_list')
    text_count = request.POST['text_count']
    text_count = int(text_count)
    text_list = text_list.split(',')
    for text in text_list:
        text = re.sub(r'\s', '', text)

    naive_corpus = []

    naive_tokenizer = Tokenizer()

    for text in text_list:
        tokens = naive_tokenizer.tokenize(text)
        element = CorpusElement(text, tokens)
        naive_corpus.append(element)

    # 感情極性対応表のロード
    pn_dic = load_pn_dict()

    # 各文章の極性値リストを得る
    for element in naive_corpus:
        #element.pn_scores = get_pn_scores(element.tokens, pn_dic)
        result = get_pn_scores_sum(element.tokens, pn_dic)
        element.pn_scores = result[0]
        element.word_count = result[1]

    show_graph(naive_corpus[text_count])
    ## min = function.find_min(naive_corpus[text_count].pn_scores)
    min = find_not_max(naive_corpus[text_count].pn_scores)
    no_count = no_word(naive_corpus)[0]

    find_text = []
    for i in range(len(min)):
        find_text.append(text_find(naive_corpus, min[i]))

    find_text_dict = {}
    for i in range(len(min)):
        find_text_dict[min[i]] = text_find(naive_corpus, min[i])

    context = {
        #'no_count' : no_count,
        'text_count' : text_count,
        'no_count_text' : no_word(naive_corpus)[1],
        'tweet_text' : tweet_text,
        'emotion_count' : find_text_dict
        #'find_text' : find_text_dict # min
    }
    return render(request, 'polls/results.html', context)

def analysis_tweet2(request):
    class CorpusElement:
        def __init__(self, text='', tokens=[], pn_scores=[], word_count=float()):
            self.text = text # テキスト本文
            self.tokens = tokens # 構文木解析されたトークンのリスト
            self.pn_scores = pn_scores # 感情極性値(後述)
            self.word_count = word_count

    tweet_text = request.POST['tweet_text']
    text_list = request.POST.get('text_list')
    text_count = request.POST['text_count']
    text_count = int(text_count)
    text_list = text_list.split(',')
    for text in text_list:
        text = re.sub(r'\s', '', text)

    naive_corpus = []

    naive_tokenizer = Tokenizer()

    for text in text_list:
        tokens = naive_tokenizer.tokenize(text)
        element = CorpusElement(text, tokens)
        naive_corpus.append(element)

    # 感情極性対応表のロード
    pn_dic = load_pn_dict()

    # 各文章の極性値リストを得る
    for element in naive_corpus:
        #element.pn_scores = get_pn_scores(element.tokens, pn_dic)
        result = get_pn_scores_sum(element.tokens, pn_dic)
        element.pn_scores = result[0]
        element.word_count = result[1]

    show_graph(naive_corpus[text_count])
    ## min = function.find_min(naive_corpus[text_count].pn_scores)
    min = find_not_max(naive_corpus[text_count].pn_scores)
    no_count = no_word(naive_corpus)[0]

    find_text = []
    for i in range(len(min)):
        find_text.append(text_find(naive_corpus, min[i]))

    find_text_dict = {}
    for i in range(len(min)):
        find_text_dict[min[i]] = text_find(naive_corpus, min[i])

    context = {
        #'no_count' : no_count,
        'text_count' : text_count,
        'tweet_text' : tweet_text,
        #'find_text' : find_text_dict # min
    }
    return render(request, 'polls/results2.html', context)


#def results(request, question_id):
#    question = get_object_or_404(Question, pk=question_id)
#    return render(request, 'polls/results.html', {'question': question})

class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    #context_object_name = 'latest_question_list'

    def get_queryset(self):
        """
        Return the last five published questions (not including those set to be
        published in the future).
        """
    #    return Question.objects.filter(
    #        pub_date__lte=timezone.now()
    #    ).order_by('-pub_date')[:5]
