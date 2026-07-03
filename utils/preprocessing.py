# import os
# import numpy as np
# from io import BytesIO
# from PIL import Image


# # --- Konstanta (harus sama dengan training) ---
# TARGET_SIZE = (224, 224)
# CLASS_NAMES = ['Grade_A', 'Grade_B', 'Grade_C']

# # Label yang lebih user-friendly untuk ditampilkan di web
# GRADE_LABELS = {
#     'Grade_A': 'Grade A (Premium)',
#     'Grade_B': 'Grade B (Standar)',
#     'Grade_C': 'Grade C (Rendah)'
# }

# # Ambang batas confidence — jika di bawah ini, dianggap bukan strawberry
# CONFIDENCE_THRESHOLD = 0.60

# # *** PERUBAHAN: MAX_FILE_SIZE_MB dari 2 menjadi 10 ***
# MAX_FILE_SIZE_MB = 10
# ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


# def preprocess_image(image_bytes):
#     """
#     Preprocessing gambar dari bytes menjadi array siap prediksi.

#     Alur:
#         bytes → PIL Image (RGB) → resize 224x224 → float32 [0, 255] → expand dims

#     *** PERUBAHAN: preprocess_input DIHAPUS ***
#     EfficientNet di TF 2.20.0 menerima input [0, 255] dan
#     melakukan normalisasi [-1, 1] secara internal melalui
#     layer Rescaling yang sudah built-in di arsitektur model.

#     Input:
#         image_bytes: bytes dari file yang diupload user

#     Output:
#         np.array shape (1, 224, 224, 3) dengan range [0, 255]
#     """
#     # Buka gambar dari bytes
#     img = Image.open(BytesIO(image_bytes)).convert('RGB')

#     # Resize ke 224x224 dengan LANCZOS (sama dengan training)
#     img = img.resize(TARGET_SIZE, Image.LANCZOS)

#     # Convert ke numpy array float32 — range [0, 255]
#     img_array = np.array(img, dtype=np.float32)


#     # Expand dims: (224,224,3) → (1,224,224,3) untuk batch size 1
#     img_array = np.expand_dims(img_array, axis=0)

#     return img_array


# def validate_file(filename, file_size_bytes):
#     """
#     Validasi format dan ukuran file sebelum diproses.

#     Input:
#         filename: nama file asli dari upload user
#         file_size_bytes: ukuran file dalam bytes

#     Output:
#         (True, None) jika valid
#         (False, "pesan error") jika tidak valid
#     """
#     # Cek ekstensi file
#     ext = os.path.splitext(filename)[1].lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         return False, f"Format file '{ext}' tidak didukung. Gunakan format JPG, JPEG, atau PNG."

#     # Cek ukuran file
#     file_size_mb = file_size_bytes / (1024 * 1024)
#     if file_size_mb > MAX_FILE_SIZE_MB:
#         return False, f"Ukuran file ({file_size_mb:.1f} MB) melebihi batas maksimal {MAX_FILE_SIZE_MB} MB."

#     return True, None


# def interpret_prediction(probabilities):
#     """
#     Interpretasi hasil prediksi model menjadi output yang siap ditampilkan.

#     Input:
#         probabilities: array shape (3,) → [prob_A, prob_B, prob_C]

#     Output:
#         dict berisi:
#         - is_strawberry: bool
#         - predicted_grade: string (nama kelas)
#         - predicted_label: string (label user-friendly)
#         - confidence: float (0.0 - 1.0)
#         - all_probabilities: dict per kelas
#     """
#     predicted_idx = np.argmax(probabilities)
#     confidence = float(probabilities[predicted_idx])
#     predicted_grade = CLASS_NAMES[predicted_idx]
#     predicted_label = GRADE_LABELS[predicted_grade]

#     all_probs = {
#         name: float(probabilities[i])
#         for i, name in enumerate(CLASS_NAMES)
#     }

#     # Cek apakah confidence cukup tinggi untuk dianggap strawberry
#     is_strawberry = confidence >= CONFIDENCE_THRESHOLD

#     return {
#         'is_strawberry': is_strawberry,
#         'predicted_grade': predicted_grade,
#         'predicted_label': predicted_label,
#         'confidence': confidence,
#         'all_probabilities': all_probs
#     }






import os
import numpy as np
from io import BytesIO
from PIL import Image


# --- Konstanta (harus sama dengan training) ---
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['Grade_A', 'Grade_B', 'Grade_C']

# Label yang lebih user-friendly untuk ditampilkan di web
GRADE_LABELS = {
    'Grade_A': 'Grade A (Premium)',
    'Grade_B': 'Grade B (Standar)',
    'Grade_C': 'Grade C (Rendah)'
}

# Ambang batas confidence — jika di bawah ini, dianggap bukan strawberry
CONFIDENCE_THRESHOLD = 0.60

MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# ============================================
# [BARU] Peringkat grade untuk logika worst-case
# Semakin tinggi angkanya, semakin buruk gradenya
# Grade_C (2) = terburuk, Grade_A (0) = terbaik
# ============================================
GRADE_RANK = {
    'Grade_A': 0,
    'Grade_B': 1,
    'Grade_C': 2
}


def preprocess_image(image_bytes):
    """
    Preprocessing gambar dari bytes menjadi array siap prediksi.

    Alur:
        bytes → PIL Image (RGB) → resize 224x224 → float32 [0, 255] → expand dims

    EfficientNet di TF 2.20.0 menerima input [0, 255] dan
    melakukan normalisasi [-1, 1] secara internal melalui
    layer Rescaling yang sudah built-in di arsitektur model.

    Input:
        image_bytes: bytes dari file yang diupload user

    Output:
        np.array shape (1, 224, 224, 3) dengan range [0, 255]
    """
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def validate_file(filename, file_size_bytes):
    """
    Validasi format dan ukuran file sebelum diproses.

    Input:
        filename: nama file asli dari upload user
        file_size_bytes: ukuran file dalam bytes

    Output:
        (True, None) jika valid
        (False, "pesan error") jika tidak valid
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Format file '{ext}' tidak didukung. Gunakan format JPG, JPEG, atau PNG."

    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"Ukuran file ({file_size_mb:.1f} MB) melebihi batas maksimal {MAX_FILE_SIZE_MB} MB."

    return True, None


def interpret_prediction(probabilities):
    """
    Interpretasi hasil prediksi model menjadi output yang siap ditampilkan.

    Input:
        probabilities: array shape (3,) → [prob_A, prob_B, prob_C]

    Output:
        dict berisi:
        - is_strawberry: bool
        - predicted_grade: string (nama kelas)
        - predicted_label: string (label user-friendly)
        - confidence: float (0.0 - 1.0)
        - all_probabilities: dict per kelas
    """
    predicted_idx = np.argmax(probabilities)
    confidence = float(probabilities[predicted_idx])
    predicted_grade = CLASS_NAMES[predicted_idx]
    predicted_label = GRADE_LABELS[predicted_grade]

    all_probs = {
        name: float(probabilities[i])
        for i, name in enumerate(CLASS_NAMES)
    }

    is_strawberry = confidence >= CONFIDENCE_THRESHOLD

    return {
        'is_strawberry': is_strawberry,
        'predicted_grade': predicted_grade,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'all_probabilities': all_probs
    }


# ============================================
# [BARU] Fungsi agregasi multi-view (worst-case grading)
# ============================================
def aggregate_multiview(results_per_sisi):
    """
    Menentukan grade akhir dari 4 hasil prediksi per sisi
    menggunakan strategi worst-case grading.

    Logika: grade akhir = grade terburuk dari semua sisi.
    Alasannya: satu sisi yang cacat (Grade C) sudah cukup
    untuk menurunkan kualitas keseluruhan buah, sesuai
    dengan prinsip grading SNI 8026:2014 dan USDA AMS 2016.

    Input:
        results_per_sisi: list of dict, masing-masing adalah
                          output dari interpret_prediction()
                          ditambah key 'sisi' (nama sisi)

    Output:
        dict berisi:
        - final_grade: string (grade akhir terburuk)
        - final_label: string (label user-friendly grade akhir)
        - final_rank: int (0/1/2, untuk referensi)
        - worst_sisi: string (nama sisi yang menentukan grade akhir)
        - detail: list of dict per sisi
    """
    worst_rank = -1
    worst_grade = None
    worst_sisi = None
    detail = []

    for r in results_per_sisi:
        grade = r['predicted_grade']
        rank = GRADE_RANK.get(grade, 0)
        detail.append({
            'sisi': r['sisi'],
            'predicted_grade': grade,
            'predicted_label': GRADE_LABELS[grade],
            'confidence': r['confidence'],
            'all_probabilities': r['all_probabilities'],
            'is_strawberry': r['is_strawberry']
        })
        if rank > worst_rank:
            worst_rank = rank
            worst_grade = grade
            worst_sisi = r['sisi']

    return {
        'final_grade': worst_grade,
        'final_label': GRADE_LABELS.get(worst_grade, ''),
        'final_rank': worst_rank,
        'worst_sisi': worst_sisi,
        'detail': detail
    }