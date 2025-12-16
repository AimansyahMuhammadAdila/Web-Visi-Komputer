from flask import Flask, render_template, request, jsonify, redirect
import numpy as np
from PIL import Image
import io
import base64
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

# ==================== GLCM FUNCTIONS (MINGGU 1) ====================
def calculate_glcm_features(image_data, angle):
    """Calculate GLCM features for given image and angle"""
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('L')
        img_array = np.array(image)
        
        angle_map = {
            0: [0],
            45: [np.pi/4],
            90: [np.pi/2],
            135: [3*np.pi/4]
        }
        angles = angle_map.get(angle, [0])
        distances = [1]
        
        glcm = graycomatrix(img_array, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        
        return {
            'contrast': f'{contrast:.6f}',
            'dissimilarity': f'{dissimilarity:.6f}',
            'homogeneity': f'{homogeneity:.6f}',
            'energy': f'{energy:.6f}',
            'correlation': f'{correlation:.6f}',
            'asm': f'{asm:.6f}'
        }
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

# ==================== KNN DATA & FUNCTIONS (MINGGU 2) ====================
training_data = {
    'luas_panen_norm': [0.45, 0.52, 0.48, 0.55, 0.50, 0.47, 0.53, 0.49, 0.51, 0.46, 0.54, 0.48,
                        0.50, 0.52, 0.49, 0.53, 0.48, 0.51, 0.47, 0.54, 0.50, 0.49, 0.52, 0.48],
    'hasil_produksi_norm': [0.42, 0.48, 0.45, 0.51, 0.47, 0.44, 0.50, 0.46, 0.49, 0.43, 0.52, 0.45,
                            0.47, 0.49, 0.46, 0.50, 0.45, 0.48, 0.44, 0.51, 0.47, 0.46, 0.49, 0.45],
    'harga': [10500, 10800, 10600, 11000, 10700, 10550, 10900, 10650, 10750, 10520, 11100, 10600,
              10700, 10800, 10650, 10900, 10600, 10750, 10550, 11000, 10700, 10650, 10800, 10600]
}

testing_data = {
    'luas_panen_norm': [0.50, 0.49, 0.52, 0.48, 0.51, 0.47, 0.53, 0.50, 0.49, 0.52, 0.48, 0.51],
    'hasil_produksi_norm': [0.47, 0.46, 0.49, 0.45, 0.48, 0.44, 0.50, 0.47, 0.46, 0.49, 0.45, 0.48],
    'harga': [10700, 10650, 10800, 10600, 10750, 10550, 10900, 10700, 10650, 10800, 10600, 10750]
}

LUAS_PANEN_MIN, LUAS_PANEN_MAX = 1000, 5000
HASIL_PRODUKSI_MIN, HASIL_PRODUKSI_MAX = 2000, 8000

def normalize_input(luas_panen, hasil_produksi):
    luas_panen_norm = (luas_panen - LUAS_PANEN_MIN) / (LUAS_PANEN_MAX - LUAS_PANEN_MIN)
    hasil_produksi_norm = (hasil_produksi - HASIL_PRODUKSI_MIN) / (HASIL_PRODUKSI_MAX - HASIL_PRODUKSI_MIN)
    return max(0, min(1, luas_panen_norm)), max(0, min(1, hasil_produksi_norm))

# ==================== PAGE ROUTES ====================
@app.route('/')
def home():
    return redirect('/tugas1')

@app.route('/tugas1')
def tugas1():
    return render_template('tugas1.html')

@app.route('/tugas2')
def tugas2():
    return render_template('tugas2.html')

@app.route('/tugas3')
def tugas3():
    return render_template('tugas3.html')

# ==================== API ROUTES ====================
@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        image_data = data.get('image')
        angle = int(data.get('angle', 0))
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        features = calculate_glcm_features(image_data, angle)
        return jsonify({'angle': angle, 'features': features})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    try:
        data = request.get_json()
        luas_panen = float(data.get('luas_panen', 0))
        hasil_produksi = float(data.get('hasil_produksi', 0))
        k = int(data.get('k', 2))
        features = data.get('features', ['luas_panen', 'hasil_produksi'])
        
        if luas_panen <= 0 or hasil_produksi <= 0:
            return jsonify({'error': 'Luas Panen dan Hasil Produksi harus lebih dari 0'}), 400
        if k < 1 or k > 10:
            return jsonify({'error': 'Nilai K harus antara 1 dan 10'}), 400
        if not features:
            return jsonify({'error': 'Pilih minimal satu fitur'}), 400
        
        df_train = pd.DataFrame(training_data)
        luas_panen_norm, hasil_produksi_norm = normalize_input(luas_panen, hasil_produksi)
        
        feature_map = {'luas_panen': 'luas_panen_norm', 'hasil_produksi': 'hasil_produksi_norm'}
        selected_columns = [feature_map[f] for f in features if f in feature_map]
        if not selected_columns:
            return jsonify({'error': 'Fitur yang dipilih tidak valid'}), 400
        
        X_train = df_train[selected_columns].values
        y_train = df_train['harga'].values
        
        input_values = []
        if 'luas_panen' in features:
            input_values.append(luas_panen_norm)
        if 'hasil_produksi' in features:
            input_values.append(hasil_produksi_norm)
        X_input = np.array([input_values])
        
        if k > len(X_train):
            k = len(X_train)
        
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        predicted_price = model.predict(X_input)[0]
        
        distances, indices = model.kneighbors(X_input)
        nearest_neighbors = []
        for i, idx in enumerate(indices[0]):
            nearest_neighbors.append({
                'index': int(idx) + 1,
                'distance': float(distances[0][i]),
                'price': float(y_train[idx])
            })
        
        neighbor_prices = [n['price'] for n in nearest_neighbors]
        
        return jsonify({
            'success': True,
            'predicted_price': round(float(predicted_price), 2),
            'k_value': k,
            'features_used': features,
            'input_original': {'luas_panen': float(luas_panen), 'hasil_produksi': float(hasil_produksi)},
            'input_normalized': {
                'luas_panen': round(float(luas_panen_norm), 4) if 'luas_panen' in features else None,
                'hasil_produksi': round(float(hasil_produksi_norm), 4) if 'hasil_produksi' in features else None
            },
            'nearest_neighbors': nearest_neighbors,
            'statistics': {
                'avg_neighbor_price': round(float(np.mean(neighbor_prices)), 2),
                'min_neighbor_price': round(float(np.min(neighbor_prices)), 2),
                'max_neighbor_price': round(float(np.max(neighbor_prices)), 2)
            }
        })
    except ValueError as ve:
        return jsonify({'error': f'Input tidak valid: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/predict_naive_bayes', methods=['POST'])
def predict_naive_bayes():
    try:
        data = request.get_json()
        csv_data = data.get('csv_data')
        test_size = float(data.get('test_size', 0.3))
        
        if not csv_data:
            return jsonify({'error': 'Tidak ada data CSV yang diberikan'}), 400
        
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data))
        
        if len(df.columns) < 2:
            return jsonify({'error': 'Dataset harus memiliki minimal 2 kolom'}), 400
        if len(df) < 10:
            return jsonify({'error': 'Dataset harus memiliki minimal 10 baris'}), 400
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        if not np.issubdtype(X.dtype, np.number):
            try:
                X = X.astype(float)
            except:
                return jsonify({'error': 'Semua kolom fitur harus numerik'}), 400
        
        try:
            y = y.astype(int)
        except:
            return jsonify({'error': 'Kolom target harus berupa class labels'}), 400
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        classes = np.unique(y).tolist()
        
        class_metrics = []
        for cls in classes:
            if str(cls) in report:
                class_metrics.append({
                    'class': int(cls),
                    'precision': round(float(report[str(cls)]['precision']), 4),
                    'recall': round(float(report[str(cls)]['recall']), 4),
                    'f1_score': round(float(report[str(cls)]['f1-score']), 4),
                    'support': int(report[str(cls)]['support'])
                })
        
        return jsonify({
            'success': True,
            'accuracy': round(float(accuracy), 4),
            'confusion_matrix': conf_matrix.tolist(),
            'classes': classes,
            'class_metrics': class_metrics,
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_features': X.shape[1],
                'num_classes': len(classes),
                'feature_names': df.columns[:-1].tolist(),
                'target_name': df.columns[-1],
                'test_size_ratio': test_size
            },
            'macro_avg': {
                'precision': round(float(report['macro avg']['precision']), 4),
                'recall': round(float(report['macro avg']['recall']), 4),
                'f1_score': round(float(report['macro avg']['f1-score']), 4)
            },
            'weighted_avg': {
                'precision': round(float(report['weighted avg']['precision']), 4),
                'recall': round(float(report['weighted avg']['recall']), 4),
                'f1_score': round(float(report['weighted avg']['f1-score']), 4)
            }
        })
    except ValueError as ve:
        return jsonify({'error': f'Input tidak valid: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
