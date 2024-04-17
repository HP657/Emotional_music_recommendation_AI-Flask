# from gensim.models import Word2Vec
# import pandas as pd
# df_train = pd.read_csv("selected_data.csv")

# # 모델 생성
# model = Word2Vec(df_train['token'], vector_size=1000, window=5, min_count=1, workers=4, sg=1)

# # 모델을 저장할 경로 결정
# model_path = "word2vec_model.bin"

# # 모델 저장
# model.save(model_path)

import pandas as pd
import numpy as np
from gensim.models import Word2Vec

def get_sent_embedding(model, embedding_size, tokenized_words):
    feature_vec = np.zeros((embedding_size,), dtype="float32")
    n_words = 0
    for word in tokenized_words:
        if word in model.wv.key_to_index:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec
df_train = pd.read_csv('selected_data.csv')
from sklearn.svm import SVC
from joblib import load, dump

model = Word2Vec.load("word2vec_model.bin")

X = [get_sent_embedding(model, 1000, tokens) for tokens in df_train['token']]
y = df_train['Emotion']

# SVM 분류 모델 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

# 모델 저장
dump(svm_model, 'svm_model.joblib')