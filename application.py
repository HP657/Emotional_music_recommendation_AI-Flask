from flask import Flask, request, jsonify
from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.svm import SVC
from joblib import load
from flask_cors import CORS
import sys
import requests
import os
import random  # 랜덤 모듈 추가

app = Flask(__name__)
CORS(app)

# Okt 형태소 분석기 초기화
okt = Okt()

# Word2Vec 모델 불러오기
model = Word2Vec.load("model/word2vec_model.bin")

# SVM 모델 불러오기
svm_model = load("model/svm_model.joblib")

svm_model = load("model/svm_model.joblib")

# Last.fm API key
last_fm_api_key = os.getenv("song_api")

# 형태소 분석 함수 정의
def tokenize(text):
    return okt.morphs(text)

# 형태소를 Word2Vec 임베딩으로 변환하는 함수 정의
def get_embedding(model, tokens):
    embeddings = [model.wv[token] for token in tokens if token in model.wv.key_to_index]
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return []

# Last.fm에서 감정에 맞는 곡을 검색하는 함수
def search_tracks_by_emotion(emotion):
    url = f"http://ws.audioscrobbler.com/2.0/?method=track.search&track={emotion}&api_key={last_fm_api_key}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        tracks = data['results']['trackmatches']['track']
        if tracks:
            random_track = random.choice(tracks)  # 랜덤한 트랙 선택
            return random_track['name'], random_track['artist']
    return None, None

@app.route('/recommend_song', methods=['POST'])
def recommend_song():
    data = request.json
    sentence = data['sentence']
    
    # 형태소로 분할
    tokenized_sentence = tokenize(sentence)
    
    # 문장을 Word2Vec 임베딩으로 변환
    embedding = get_embedding(model, tokenized_sentence)
    
    # 감정 분류
    emotion = svm_model.predict([embedding])[0]
    print(emotion)
    # 감정에 따른 곡 추천
    track_name, artist = search_tracks_by_emotion(emotion)
    print(track_name, artist)
    if track_name and artist:
        return jsonify({"emotion": emotion, "recommended_track": f"{track_name} by {artist}"})
    else:
        return jsonify({"emotion": emotion, "recommended_track": "No track found for the specified emotion"})

if __name__ == '__main__':
    app.run(port=int(sys.argv[1]), host="0.0.0.0", debug=True)
