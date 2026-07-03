# import os
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.optimizers import Adam
# from utils.preprocessing import preprocess_image, validate_file, interpret_prediction

# # KONFIGURASI

# app = Flask(
#     __name__,
#     template_folder='templates',
#     static_folder='static'
# )

# app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# MODEL_PATH = os.path.join('model', 'strawberry_grading_final.keras')



# # FUNGSI: Bangun ulang arsitektur model

# def build_model():
#     """
#     Bangun ulang arsitektur yang sama persis dengan saat training.
#     Arsitektur: EfficientNetB0 → GAP → Dense(128,ReLU) → Dropout(0.5) → Softmax(3)

#     FIX: Menggunakan weights='imagenet' agar arsitektur dan
#     BatchNormalization statistics identik dengan saat training.
#     load_weights() kemudian menimpa semua bobot termasuk head.
#     ImageNet weights hanya di-download sekali dan di-cache.
#     """
#     base_model = EfficientNetB0(
#         weights='imagenet',         # FIX: pakai imagenet, bukan None
#         include_top=False,
#         input_shape=(224, 224, 3)
#     )
#     base_model.trainable = False

#     inputs = Input(shape=(224, 224, 3))
#     x = base_model(inputs, training=False)
#     x = GlobalAveragePooling2D(name='gap')(x)
#     x = Dense(128, activation='relu', name='dense_128')(x)
#     x = Dropout(0.5, name='dropout_05')(x)
#     outputs = Dense(3, activation='softmax', name='output_softmax')(x)

#     model = Model(inputs=inputs, outputs=outputs, name='StrawberryGrading_EfficientNetB0')
#     return model


# # ============================================
# # LOAD MODEL
# # ============================================
# print(f"🔄 Loading model dari: {MODEL_PATH}")
# try:
#     model = build_model()
#     model.load_weights(MODEL_PATH)
#     model.compile(
#         optimizer=Adam(learning_rate=0.0001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     print("✅ Model berhasil di-load! (build_model + load_weights)")
# except Exception as e:
#     print(f"❌ Gagal load model: {e}")
#     print("→ Pastikan file model ada di folder: model/")
#     print(f"→ Expected path: {MODEL_PATH}")
#     model = None


# # ============================================
# # ROUTE: Halaman Utama
# # ============================================
# @app.route('/')
# def index():
#     return render_template('index.html')


# # ============================================
# # ROUTE: Halaman Klasifikasi
# # ============================================
# @app.route('/klasifikasi')
# def klasifikasi():
#     return render_template('klasifikasi.html')


# # ============================================
# # ROUTE: API Prediksi
# # ============================================
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if model is None:
#             return jsonify({
#                 'success': False,
#                 'error': 'Model belum berhasil di-load. Cek log terminal.'
#             }), 500

#         if 'file' not in request.files:
#             return jsonify({
#                 'success': False,
#                 'error': 'Tidak ada file yang diupload. Silakan pilih gambar terlebih dahulu.'
#             }), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({
#                 'success': False,
#                 'error': 'Tidak ada file yang dipilih. Silakan pilih gambar terlebih dahulu.'
#             }), 400

#         file_bytes = file.read()
#         is_valid, error_msg = validate_file(file.filename, len(file_bytes))
#         if not is_valid:
#             return jsonify({'success': False, 'error': error_msg}), 400

#         try:
#             img_array = preprocess_image(file_bytes)
#         except Exception:
#             return jsonify({
#                 'success': False,
#                 'error': 'File tidak dapat dibaca sebagai gambar. Pastikan file adalah gambar yang valid.'
#             }), 400

#         predictions = model.predict(img_array, verbose=0)
#         probabilities = predictions[0]

#         result = interpret_prediction(probabilities)

#         if not result['is_strawberry']:
#             return jsonify({
#                 'success': True,
#                 'is_strawberry': False,
#                 'message': 'Gambar tidak terdeteksi sebagai buah strawberry. Silakan upload foto buah strawberry.'
#             })

#         return jsonify({
#             'success': True,
#             'is_strawberry': True,
#             'predicted_grade': result['predicted_grade'],
#             'confidence': round(result['confidence'] * 100, 2),
#             'probabilities': {
#                 'Grade_A': round(result['all_probabilities']['Grade_A'] * 100, 2),
#                 'Grade_B': round(result['all_probabilities']['Grade_B'] * 100, 2),
#                 'Grade_C': round(result['all_probabilities']['Grade_C'] * 100, 2),
#             }
#         })

#     except Exception as e:
#         return jsonify({
#             'success': False,
#             'error': f'Terjadi kesalahan pada server: {str(e)}'
#         }), 500


# # ============================================
# # ERROR HANDLER: File terlalu besar
# # ============================================
# @app.errorhandler(413)
# def file_too_large(e):
#     return jsonify({
#         'success': False,
#         'error': 'Ukuran file melebihi batas maksimal 10 MB.'
#     }), 413


# # ============================================
# # JALANKAN SERVER
# # ============================================
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)








import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from utils.preprocessing import (
    preprocess_image,
    validate_file,
    interpret_prediction,
    aggregate_multiview   # [BARU] fungsi agregasi multi-view
)

# ============================================
# KONFIGURASI
# ============================================
app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)

app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024  # 10 MB

MODEL_PATH = os.path.join('model', 'strawberry_grading_final.keras')

# [BARU] Nama-nama field file yang dikirim dari frontend
# Harus cocok persis dengan name= di FormData JavaScript
SISI_FIELDS = ['sisi_depan', 'sisi_kanan', 'sisi_kiri', 'sisi_belakang']
SISI_LABELS = {
    'sisi_depan':   'Depan',
    'sisi_kanan':   'Kanan',
    'sisi_kiri':    'Kiri',
    'sisi_belakang': 'Belakang'
}


# ============================================
# FUNGSI: Bangun ulang arsitektur model
# ============================================
def build_model():
    """
    Bangun ulang arsitektur yang sama persis dengan saat training.
    Arsitektur: EfficientNetB0 → GAP → Dense(128,ReLU) → Dropout(0.5) → Softmax(3)

    Menggunakan weights='imagenet' agar BatchNormalization statistics
    identik dengan saat training. load_weights() kemudian menimpa
    semua bobot termasuk head.
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = Dropout(0.5, name='dropout_05')(x)
    outputs = Dense(3, activation='softmax', name='output_softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='StrawberryGrading_EfficientNetB0')
    return model


# ============================================
# LOAD MODEL
# ============================================
print(f"🔄 Loading model dari: {MODEL_PATH}")
try:
    model = build_model()
    model.load_weights(MODEL_PATH)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Model berhasil di-load! (build_model + load_weights)")
except Exception as e:
    print(f"❌ Gagal load model: {e}")
    print("→ Pastikan file model ada di folder: model/")
    print(f"→ Expected path: {MODEL_PATH}")
    model = None


# ============================================
# ROUTE: Halaman Utama
# ============================================
@app.route('/')
def index():
    return render_template('index.html')


# ============================================
# ROUTE: Halaman Klasifikasi
# ============================================
@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')


# ============================================
# ROUTE: API Prediksi Multi-View (4 sisi)
# [DIUBAH] dari 1 file menjadi 4 file sekaligus
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cek model sudah ter-load
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model belum berhasil di-load. Cek log terminal.'
            }), 500

        # -------------------------------------------
        # [BARU] Validasi: semua 4 sisi harus ada
        # -------------------------------------------
        missing_sisi = [s for s in SISI_FIELDS if s not in request.files]
        if missing_sisi:
            label_missing = [SISI_LABELS[s] for s in missing_sisi]
            return jsonify({
                'success': False,
                'error': f"Foto sisi {', '.join(label_missing)} belum diupload. "
                         f"Pastikan keempat sisi sudah dipilih sebelum mengklik Klasifikasi."
            }), 400

        # -------------------------------------------
        # [BARU] Proses setiap sisi satu per satu,
        # kumpulkan hasilnya ke list
        # -------------------------------------------
        results_per_sisi = []

        for field in SISI_FIELDS:
            file = request.files[field]
            sisi_label = SISI_LABELS[field]

            # Cek file tidak kosong
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': f"Foto sisi {sisi_label} kosong. Silakan pilih file terlebih dahulu."
                }), 400

            # Baca bytes
            file_bytes = file.read()

            # Validasi format & ukuran
            is_valid, error_msg = validate_file(file.filename, len(file_bytes))
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': f"Sisi {sisi_label}: {error_msg}"
                }), 400

            # Preprocessing
            try:
                img_array = preprocess_image(file_bytes)
            except Exception:
                return jsonify({
                    'success': False,
                    'error': f"Foto sisi {sisi_label} tidak dapat dibaca. "
                             f"Pastikan file adalah gambar yang valid (JPG/PNG)."
                }), 400

            # Prediksi
            predictions = model.predict(img_array, verbose=0)
            probabilities = predictions[0]

            # Interpretasi hasil
            result = interpret_prediction(probabilities)
            result['sisi'] = sisi_label   # tambahkan label sisi ke result

            results_per_sisi.append(result)

        # -------------------------------------------
        # [BARU] Cek: minimal 1 sisi harus terdeteksi
        # sebagai strawberry (confidence >= threshold)
        # -------------------------------------------
        sisi_bukan_strawberry = [
            r['sisi'] for r in results_per_sisi if not r['is_strawberry']
        ]

        # Kalau SEMUA sisi tidak terdeteksi sebagai strawberry → tolak
        if len(sisi_bukan_strawberry) == len(SISI_FIELDS):
            return jsonify({
                'success': True,
                'is_strawberry': False,
                'message': 'Tidak ada sisi yang terdeteksi sebagai buah stroberi. '
                           'Pastikan foto menampilkan buah stroberi dengan jelas.'
            })

        # Kalau SEBAGIAN sisi tidak terdeteksi → tetap lanjut tapi tandai
        # (sisi yang tidak terdeteksi tetap ikut worst-case dengan grade yang diprediksi)

        # -------------------------------------------
        # [BARU] Agregasi worst-case grading
        # -------------------------------------------
        agregasi = aggregate_multiview(results_per_sisi)

        # -------------------------------------------
        # [BARU] Susun response lengkap
        # -------------------------------------------
        detail_per_sisi = []
        for d in agregasi['detail']:
            detail_per_sisi.append({
                'sisi': d['sisi'],
                'predicted_grade': d['predicted_grade'],
                'predicted_label': d['predicted_label'],
                'confidence': round(d['confidence'] * 100, 2),
                'is_strawberry': d['is_strawberry'],
                'probabilities': {
                    'Grade_A': round(d['all_probabilities']['Grade_A'] * 100, 2),
                    'Grade_B': round(d['all_probabilities']['Grade_B'] * 100, 2),
                    'Grade_C': round(d['all_probabilities']['Grade_C'] * 100, 2),
                }
            })

        return jsonify({
            'success': True,
            'is_strawberry': True,

            # Grade akhir (hasil agregasi worst-case)
            'final_grade': agregasi['final_grade'],
            'final_label': agregasi['final_label'],
            'worst_sisi': agregasi['worst_sisi'],

            # Detail per sisi (untuk ditampilkan di tabel)
            'detail_per_sisi': detail_per_sisi,

            # Sisi yang confidence-nya rendah (untuk peringatan di UI)
            'sisi_confidence_rendah': sisi_bukan_strawberry
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Terjadi kesalahan pada server: {str(e)}'
        }), 500


# ============================================
# ERROR HANDLER: File terlalu besar
# ============================================
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        'success': False,
        'error': 'Ukuran file melebihi batas maksimal 10 MB.'
    }), 413


# ============================================
# JALANKAN SERVER
# ============================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)