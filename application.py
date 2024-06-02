from flask import Flask, request, jsonify
from konlpy.tag import Okt
from gensim.models import Word2Vec
from sklearn.svm import SVC
from joblib import load
from flask_cors import CORS
import sys
import requests
import os
import random  

app = Flask(__name__)
CORS(app)

okt = Okt()
model = Word2Vec.load("model/word2vec_model.bin")
svm_model = load("model/svm_model.joblib")
last_fm_api_key = os.getenv("song_api")

def tokenize(text):
    return okt.morphs(text)

def get_embedding(model, tokens):
    embeddings = [model.wv[token] for token in tokens if token in model.wv.key_to_index]
    if embeddings:
        return sum(embeddings) / len(embeddings)
    else:
        return [0] * model.vector_size  

def get_song_info(artist, track):
    url = f'http://musicbrainz.org/ws/2/recording?query=artist:"{artist}" AND recording:"{track}"&fmt=json'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'recordings' in data and len(data['recordings']) > 0:
            recording_info = data['recordings'][0]
            if 'releases' in recording_info and len(recording_info['releases']) > 0:
                album_mbid = recording_info['releases'][0]['id']
                return album_mbid
    return None

def get_album_cover(album_mbid):
    if not album_mbid:
        return None
    url = f'http://coverartarchive.org/release/{album_mbid}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'images' in data and len(data['images']) > 0:
            image_url = data['images'][0]['image']
            return image_url
    return None

def search_tracks_by_emotion(emotion):
    url = f"http://ws.audioscrobbler.com/2.0/?method=track.search&track={emotion}&api_key={last_fm_api_key}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        tracks = data['results']['trackmatches']['track']
        if tracks:
            random_track = random.choice(tracks)  
            return random_track['name'], random_track['artist']
    return None, None

@app.route('/recommend_song', methods=['POST'])
def recommend_song():
    data = request.json
    sentence = data.get('sentence')
    if not sentence:
        return jsonify({"error": "Sentence is missing."}), 400
    
    tokenized_sentence = tokenize(sentence)
    embedding = get_embedding(model, tokenized_sentence)
    emotion = svm_model.predict([embedding])[0]
    
    track_name, artist = search_tracks_by_emotion(emotion)
    
    if track_name and artist:
        album_mbid = get_song_info(artist, track_name)
        if album_mbid:
            album_cover_url = get_album_cover(album_mbid)
            return jsonify({"emotion": emotion, 
                            "recommended_track": track_name,
                            "recommended_artist": artist,
                            "album_cover_url": album_cover_url or "Album cover not found"})
        else:
            return jsonify({"emotion": emotion,
                            "recommended_track": track_name,
                            "recommended_artist": artist,
                            "album_cover_url": "Album cover not found"})
    else:
        return jsonify({"error": "No recommended track found."}), 404

if __name__ == '__main__':
    app.run(port=int(sys.argv[1]), host="0.0.0.0", debug=False)
