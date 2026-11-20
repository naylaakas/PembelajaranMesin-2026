import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load Model, Scaler, dan TF-IDF Vectorizer
MODEL_PATH = 'cafe_sentiment_model.h5'
SCALER_PATH = 'scaler.pkl'
TFIDF_PATH = 'tfidf.pkl'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(TFIDF_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    print("✅ System Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading system: {e}")

def preprocess_input(cafe_name, review_text, rating_cafe, review_count, lat, lon):
    panjang_teks = len(review_text)
    teks_gabungan = f"{cafe_name} {review_text}"
    
    X_txt = tfidf.transform([teks_gabungan]).toarray()
    X_num = np.array([[float(rating_cafe), float(review_count), float(lat), float(lon), float(panjang_teks)]])
    X_feat = np.hstack([X_num, X_txt])
    
    return scaler.transform(X_feat)

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <div style="max-width:480px; margin:40px auto; font-family:sans-serif; padding:24px; border:1px solid #ddd; border-radius:10px;">
        <h2 style="text-align:center;">Prediksi Sentimen Ulasan Kafe</h2>
        <form action="/predict" method="post">
            <label>Nama Kafe:</label><br>
            <input type="text" name="cafe_name" value="Kopi Tuku" style="width:100%; margin-bottom:10px; padding:6px;" required><br>
            
            <label>Teks Ulasan:</label><br>
            <textarea name="review_text" rows="3" style="width:100%; margin-bottom:10px; padding:6px;" placeholder="Contoh: Tempatnya nyaman banget buat nugas, kopinya enak!" required></textarea><br>
            
            <label>Rating Kafe (1.0 - 5.0):</label><br>
            <input type="number" step="0.1" name="rating_cafe" value="4.7" style="width:100%; margin-bottom:10px; padding:6px;" required><br>
            
            <label>Jumlah Review Kafe:</label><br>
            <input type="number" name="review_count" value="320" style="width:100%; margin-bottom:10px; padding:6px;" required><br>
            
            <label>Latitude:</label><br>
            <input type="number" step="any" name="lat" value="-7.7956" style="width:100%; margin-bottom:10px; padding:6px;" required><br>
            
            <label>Longitude:</label><br>
            <input type="number" step="any" name="lon" value="110.3695" style="width:100%; margin-bottom:16px; padding:6px;" required><br>
            
            <button type="submit" style="width:100%; padding:10px; background:#2563eb; color:white; border:none; border-radius:6px; cursor:pointer;">Prediksi</button>
        </form>
    </div>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = preprocess_input(
            request.form['cafe_name'],
            request.form['review_text'],
            request.form['rating_cafe'],
            request.form['review_count'],
            request.form['lat'],
            request.form['lon']
        )
        prediction = float(model.predict(data)[0][0])
        label = "Positif (Rating ≥ 4)" if prediction > 0.5 else "Negatif (Rating < 4)"
        
        return f"<h2 style='text-align:center'>Hasil: {label} (Skor: {prediction:.4f})</h2><center><a href='/'>Kembali</a></center>"
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7861, debug=True)