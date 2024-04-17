from flask import Flask, request, jsonify
from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.svm import SVC
from joblib import load
from flask_cors import CORS
import sys

app = Flask(__name__)
CORS(app)
# Okt 형태소 분석기 초기화
okt = Okt()

# Word2Vec 모델 불러오기
model = Word2Vec.load("modeling/word2vec_model.bin")

# SVM 모델 불러오기
svm_model = SVC(kernel='linear')
svm_model = load("modeling/svm_model.joblib")

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
    
@app.route("/", methods=['GET'])
def main():
    return 'hello'

@app.route('/classify', methods=['POST'])
def classify_emotion():
    data = request.json
    sentence = data['sentence']

    # 형태소로 분할
    tokenized_sentence = tokenize(sentence)

    # 문장을 Word2Vec 임베딩으로 변환
    embedding = get_embedding(model, tokenized_sentence)

    # 감정 분류
    emotion = svm_model.predict([embedding])[0]

    return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(port=int(sys.argv[1]),host="0.0.0.0",debug=True)
