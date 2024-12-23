from flask import Flask, render_template, request,jsonify
import pandas as pd
import plotly.express as px
import pickle
import joblib
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model machine learning
model_path = "dropout_model.pkl"  # Pastikan model sudah disimpan sebelumnya
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Fungsi untuk membuat grafik pie chart
def generate_pie_chart():
    # Contoh data distribusi dropout dan graduate
    data = pd.DataFrame({
        'status': ['graduate', 'dropout', 'graduate', 'graduate', 'dropout'],
    })

    # Hitung jumlah graduate dan dropout
    status_counts = data['status'].value_counts()

    # Buat pie chart
    fig = px.pie(names=status_counts.index, values=status_counts.values, 
                 title="Distribusi Status Siswa (Dropout vs Graduate)",
                 labels={'label': 'Status', 'value': 'Jumlah Siswa'})

    # Simpan grafik sebagai HTML di folder static
    fig.write_html("static/pie_chart.html")

# Route untuk halaman utama
@app.route('/')
def dashboard():
    # Buat grafik pie chart
    generate_pie_chart()
    return render_template('dashboard.html')
    # return (/dashboard.html)


# Route untuk input prediksi
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Ambil data input dari form
        data = request.json
        input_data = [
            data.get('marital_status', None),
            data.get('daytime_attendance', None),
            data.get('previous_qualification', None),
            data.get('previous_qualification_grade', None),
            data.get('mothers_qualification', None),
            data.get('fathers_qualification', None),
            data.get('mothers_occupation', None),
            data.get('fathers_occupation', None),
            data.get('admission_grade', None),
            data.get('displaced', None),
            data.get('educational_special_needs', None),
            data.get('debtor', None),
            data.get('tuition_fees_update', None),
            data.get('gender', None),
            data.get('scholarship_holder', None),
            data.get('age', None),
            data.get('international', None),
            data.get('sem_one_grade', None),
            data.get('sem_two_grade', None)
        ]

        input_df = pd.DataFrame([input_data], columns=[
            'marital_status','daytime_attendance','previous_qualification',
            'previous_qualification_grade','mothers_qualification',
            'fathers_qualification','mothers_occupation','fathers_occupation',
            'admission_grade','displaced','educational_special_needs','debtor',
            'tuition_fees_update','gender','scholarship_holder','age',
            'international','sem_one_grade','sem_two_grade'
        ])

        # Prediksi dengan model
        model = joblib.load("dropout_model.pkl")
        print(type(model))
        prediction = model.predict(input_df)[0]
        result = "Graduate" if prediction == 0 else "Dropout"

        # Tampilkan hasil
        return render_template('predict.html', result=result, input_data=input_data)
    
    # Jika GET, tampilkan form input
    return render_template('predict.html', result=None)

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
