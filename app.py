from flask import Flask, request, render_template, session
import joblib
import pandas as pd
import re
import string
import nltk  
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from datetime import datetime
import os

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load model & data
model = joblib.load("model_hukuman_kekerasan_seksual.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")
data = pd.read_csv("kasus_seksual(4).csv", sep=';')

# NLP tools
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return stemmer.stem(' '.join(words))

@app.route("/")
def home():
    return render_template("index.html", page="home", current_year=datetime.now().year)

@app.route("/pasal")
def pasal():
    return render_template("index.html", page="pasal", current_year=datetime.now().year)

@app.route("/konsultasi", methods=["GET", "POST"])
def konsultasi():
    teks = ""
    rekomendasi = penjelasan = penjara = denda = None

    if request.method == "POST":
        teks = request.form.get("teks_kasus", "")
        teks_clean = clean_text(teks)
        tfidf = vectorizer.transform([teks_clean])
        pred = model.predict(tfidf)[0]
        rekomendasi = le.inverse_transform([pred])[0]

        # Cari data hukum terkait
        filtered_data = data[data["Respon_Hukum"] == rekomendasi]
        if not filtered_data.empty:
            hasil = filtered_data.iloc[0]
            penjelasan = hasil.get("Penjelasan_pasal", "Tidak tersedia")
            penjara = hasil.get("Hukuman_Penjara", "Tidak tersedia")
            denda = hasil.get("Denda", "Tidak tersedia")
        else:
            penjelasan = "Tidak ditemukan."
            penjara = denda = "Tidak tersedia."

    return render_template(
        "index.html",
        page="konsultasi",
        teks=teks,
        rekomendasi=rekomendasi,
        penjelasan=penjelasan,
        penjara=penjara,
        denda=denda,
        current_year=datetime.now().year
    )

if __name__ == "__main__":
    app.run(debug=True)
