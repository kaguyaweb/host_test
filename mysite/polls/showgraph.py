#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

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
    plt.savefig('out.png')