import io

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


def find_text(texts, min):
    element = sorted(texts, key=lambda e: e.pn_scores[min]/e.word_count, reverse=True)
    return io.StringIO(element[0].text).read()

def no_count(texts):
    count = 0
    count_index = []
    for i in range(len(texts)):
        if texts[i].word_count == 0.0001:
            count += 1
            count_index.append(i)
    
    return count, count_index