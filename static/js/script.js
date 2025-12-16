// ==================== GLOBAL VARIABLES ====================
let uploadedImage = null;
let selectedAngle = 0;
let uploadedCSVData = null;
let csvFileName = '';

// ==================== GLCM FUNCTIONS ====================
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            uploadedImage = e.target.result;
            const preview = document.getElementById('previewImage');
            preview.src = e.target.result;
            preview.style.display = 'block';
            document.getElementById('angleCard').style.display = 'block';
            document.getElementById('resultsCard').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
}

function selectAngle(angle) {
    selectedAngle = angle;
    document.querySelectorAll('.angle-button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
}

async function processImage() {
    if (!uploadedImage) {
        alert('Silakan upload gambar terlebih dahulu!');
        return;
    }

    document.getElementById('resultsCard').style.display = 'block';
    document.getElementById('loading').classList.add('show');
    document.getElementById('resultsContent').innerHTML = '';

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: uploadedImage,
                angle: selectedAngle
            })
        });

        const data = await response.json();
        document.getElementById('loading').classList.remove('show');

        if (data.error) {
            document.getElementById('resultsContent').innerHTML =
                `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            displayGLCMResults(data);
        }
    } catch (error) {
        document.getElementById('loading').classList.remove('show');
        document.getElementById('resultsContent').innerHTML =
            `<p style="color: red;">Error: ${error.message}</p>`;
    }
}

function displayGLCMResults(data) {
    const html = `
        <p style="margin-bottom: 10px; color: #667eea; font-weight: 600;">
            Sudut: ${data.angle}°
        </p>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Fitur</th>
                    <th>Nilai</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Contrast</td><td>${data.features.contrast}</td></tr>
                <tr><td>Dissimilarity</td><td>${data.features.dissimilarity}</td></tr>
                <tr><td>Homogeneity</td><td>${data.features.homogeneity}</td></tr>
                <tr><td>Energy</td><td>${data.features.energy}</td></tr>
                <tr><td>ASM</td><td>${data.features.asm}</td></tr>
                <tr><td>Correlation</td><td>${data.features.correlation}</td></tr>
            </tbody>
        </table>
    `;
    document.getElementById('resultsContent').innerHTML = html;
}

// ==================== KNN FUNCTIONS ====================
async function processKNN(event) {
    event.preventDefault();

    const loadingKNN = document.getElementById('loadingKNN');
    const resultKNN = document.getElementById('resultKNN');

    loadingKNN.style.display = 'block';
    resultKNN.style.display = 'none';

    const luasPanen = parseFloat(document.getElementById('luas_panen').value);
    const hasilProduksi = parseFloat(document.getElementById('hasil_produksi').value);
    const nilaiK = parseInt(document.getElementById('nilai_k').value);

    const featureCheckboxes = document.querySelectorAll('input[name="features"]:checked');
    const selectedFeatures = Array.from(featureCheckboxes).map(cb => cb.value);

    if (selectedFeatures.length === 0) {
        alert('⚠️ Pilih minimal satu fitur!');
        loadingKNN.style.display = 'none';
        return;
    }

    try {
        const response = await fetch('/predict_knn', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                luas_panen: luasPanen,
                hasil_produksi: hasilProduksi,
                k: nilaiK,
                features: selectedFeatures
            })
        });

        const data = await response.json();

        if (data.success) {
            displayKNNResults(data);
            resultKNN.style.display = 'block';
        } else {
            alert('❌ Error: ' + data.error);
        }
    } catch (error) {
        alert('❌ Terjadi kesalahan: ' + error.message);
    } finally {
        loadingKNN.style.display = 'none';
    }
}

function displayKNNResults(data) {
    const resultContent = document.getElementById('knnResultContent');

    let resultHTML = `
        <div class="prediction-result-wrapper">
            <div class="prediction-main-card">
                <div class="prediction-icon">💰</div>
                <h4>Harga Prediksi Beras</h4>
                <div class="price-prediction">
                    Rp ${data.predicted_price.toLocaleString('id-ID')}
                </div>
                <p class="prediction-note">per kilogram</p>
            </div>
            
            <div class="detail-section">
                <h4 class="section-title">📋 Detail Input</h4>
                <div class="detail-grid">
                    <div class="detail-card">
                        <span class="detail-icon">🌱</span>
                        <div class="detail-info">
                            <span class="detail-label">Luas Panen</span>
                            <span class="detail-value">${data.input_original.luas_panen.toLocaleString('id-ID')} ha</span>
                        </div>
                    </div>
                    <div class="detail-card">
                        <span class="detail-icon">📦</span>
                        <div class="detail-info">
                            <span class="detail-label">Hasil Produksi</span>
                            <span class="detail-value">${data.input_original.hasil_produksi.toLocaleString('id-ID')} ton</span>
                        </div>
                    </div>
                    <div class="detail-card">
                        <span class="detail-icon">🎯</span>
                        <div class="detail-info">
                            <span class="detail-label">Nilai K</span>
                            <span class="detail-value">${data.k_value}</span>
                        </div>
                    </div>
                    <div class="detail-card">
                        <span class="detail-icon">✅</span>
                        <div class="detail-info">
                            <span class="detail-label">Fitur Digunakan</span>
                            <span class="detail-value">${data.features_used.length} fitur</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="detail-section">
                <h4 class="section-title">📐 Nilai Ternormalisasi</h4>
                <div class="normalized-grid">
    `;

    if (data.input_normalized.luas_panen !== null) {
        resultHTML += `
            <div class="normalized-item">
                <span class="norm-label">Luas Panen:</span>
                <div class="norm-bar-container">
                    <div class="norm-bar" style="width: ${data.input_normalized.luas_panen * 100}%"></div>
                </div>
                <span class="norm-value">${data.input_normalized.luas_panen}</span>
            </div>
        `;
    }

    if (data.input_normalized.hasil_produksi !== null) {
        resultHTML += `
            <div class="normalized-item">
                <span class="norm-label">Hasil Produksi:</span>
                <div class="norm-bar-container">
                    <div class="norm-bar" style="width: ${data.input_normalized.hasil_produksi * 100}%"></div>
                </div>
                <span class="norm-value">${data.input_normalized.hasil_produksi}</span>
            </div>
        `;
    }

    resultHTML += `
                </div>
            </div>
            
            <div class="detail-section">
                <h4 class="section-title">📊 Statistik K-Nearest Neighbors</h4>
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-icon">📈</span>
                        <span class="stat-label">Rata-rata</span>
                        <span class="stat-value">Rp ${data.statistics.avg_neighbor_price.toLocaleString('id-ID')}</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-icon">📉</span>
                        <span class="stat-label">Minimum</span>
                        <span class="stat-value">Rp ${data.statistics.min_neighbor_price.toLocaleString('id-ID')}</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-icon">📊</span>
                        <span class="stat-label">Maximum</span>
                        <span class="stat-value">Rp ${data.statistics.max_neighbor_price.toLocaleString('id-ID')}</span>
                    </div>
                </div>
            </div>
            
            <div class="detail-section">
                <h4 class="section-title">🎯 Data K-Nearest Neighbors</h4>
                <div class="neighbors-list">
    `;

    data.nearest_neighbors.forEach((neighbor, index) => {
        resultHTML += `
            <div class="neighbor-card">
                <div class="neighbor-rank">#${index + 1}</div>
                <div class="neighbor-info">
                    <span class="neighbor-price">Rp ${neighbor.price.toLocaleString('id-ID')}</span>
                    <span class="neighbor-distance">Jarak Euclidean: ${neighbor.distance.toFixed(4)}</span>
                </div>
            </div>
        `;
    });

    resultHTML += `
                </div>
            </div>
        </div>
    `;

    resultContent.innerHTML = resultHTML;
}

// ==================== NAIVE BAYES FUNCTIONS ====================
function handleCSVUpload(event) {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.csv')) {
        csvFileName = file.name;
        const reader = new FileReader();

        reader.onload = function (e) {
            uploadedCSVData = e.target.result;
            previewDataset(uploadedCSVData);

            const lines = uploadedCSVData.trim().split('\n');
            const columns = lines[0].split(',').length;

            document.getElementById('csvFileName').textContent = csvFileName;
            document.getElementById('csvRows').textContent = lines.length - 1;
            document.getElementById('csvCols').textContent = columns;
            document.getElementById('csvInfo').style.display = 'block';
            document.getElementById('btnProcessNB').disabled = false;
        };

        reader.readAsText(file);
    } else {
        alert('⚠️ Silakan upload file CSV yang valid');
    }
}

function previewDataset(csvData) {
    const lines = csvData.trim().split('\n');
    const header = lines[0].split(',');
    const previewRows = lines.slice(0, Math.min(11, lines.length));

    let tableHTML = '<table class="preview-table"><thead><tr>';
    header.forEach(col => {
        tableHTML += `<th>${col.trim()}</th>`;
    });
    tableHTML += '</tr></thead><tbody>';

    for (let i = 1; i < previewRows.length; i++) {
        const cells = previewRows[i].split(',');
        tableHTML += '<tr>';
        cells.forEach(cell => {
            tableHTML += `<td>${cell.trim()}</td>`;
        });
        tableHTML += '</tr>';
    }

    tableHTML += '</tbody></table>';
    document.getElementById('dataPreviewContent').innerHTML = tableHTML;
    document.getElementById('dataPreview').style.display = 'block';
}

async function processNaiveBayes() {
    if (!uploadedCSVData) {
        alert('⚠️ Silakan upload file CSV terlebih dahulu');
        return;
    }

    const loadingNB = document.getElementById('loadingNB');
    const resultNB = document.getElementById('resultNB');
    const testSize = parseFloat(document.getElementById('test_size_nb').value);

    loadingNB.style.display = 'block';
    resultNB.style.display = 'none';

    try {
        const response = await fetch('/predict_naive_bayes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                csv_data: uploadedCSVData,
                test_size: testSize
            })
        });

        const data = await response.json();

        if (data.success) {
            displayNBResults(data);
            resultNB.style.display = 'block';
        } else {
            alert('❌ Error: ' + data.error);
        }
    } catch (error) {
        alert('❌ Terjadi kesalahan: ' + error.message);
    } finally {
        loadingNB.style.display = 'none';
    }
}

function displayNBResults(data) {
    const resultContent = document.getElementById('nbResultContent');

    let html = `
        <div class="accuracy-card">
            <div class="accuracy-label">Akurasi</div>
            <div class="accuracy-value">${(data.accuracy * 100).toFixed(2)}%</div>
        </div>
        
        <div class="detail-section">
            <h4 class="section-title">📋 Informasi Dataset</h4>
            <div class="dataset-info-grid">
                <div class="info-item">
                    <div class="info-item-label">Total Sampel</div>
                    <div class="info-item-value">${data.dataset_info.total_samples}</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Sampel Training</div>
                    <div class="info-item-value">${data.dataset_info.train_samples}</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Sampel Testing</div>
                    <div class="info-item-value">${data.dataset_info.test_samples}</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Jumlah Fitur</div>
                    <div class="info-item-value">${data.dataset_info.num_features}</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Jumlah Kelas</div>
                    <div class="info-item-value">${data.dataset_info.num_classes}</div>
                </div>
                <div class="info-item">
                    <div class="info-item-label">Rasio Testing</div>
                    <div class="info-item-value">${(data.dataset_info.test_size_ratio * 100).toFixed(0)}%</div>
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <h4 class="section-title">📊 Metrik Keseluruhan</h4>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Presisi</div>
                    <div class="metric-value">${data.weighted_avg.precision.toFixed(4)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">${data.weighted_avg.recall.toFixed(4)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Skor F1</div>
                    <div class="metric-value">${data.weighted_avg.f1_score.toFixed(4)}</div>
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <h4 class="section-title">🎯 Metrik Per-Kelas</h4>
            <table class="class-metrics-table">
                <thead>
                    <tr>
                        <th>Kelas</th>
                        <th>Presisi</th>
                        <th>Recall</th>
                        <th>Skor F1</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
    `;

    data.class_metrics.forEach(metric => {
        html += `
            <tr>
                <td>${metric.class}</td>
                <td>${metric.precision.toFixed(4)}</td>
                <td>${metric.recall.toFixed(4)}</td>
                <td>${metric.f1_score.toFixed(4)}</td>
                <td>${metric.support}</td>
            </tr>
        `;
    });

    html += `
                </tbody>
            </table>
        </div>
        
        <div class="detail-section">
            <h4 class="section-title">🔢 Matriks Kebingungan</h4>
            <div class="confusion-matrix">
                <div style="text-align: center;">
                    ${drawConfusionMatrix(data.confusion_matrix, data.classes)}
                </div>
                <div class="cm-labels">
                    <strong>Baris:</strong> Kelas Aktual | <strong>Kolom:</strong> Kelas Prediksi
                </div>
            </div>
        </div>
    `;

    resultContent.innerHTML = html;
}

function drawConfusionMatrix(matrix, classes) {
    const numClasses = classes.length;
    let maxValue = 0;

    matrix.forEach(row => {
        row.forEach(val => {
            if (val > maxValue) maxValue = val;
        });
    });

    let html = `<div class="cm-grid" style="grid-template-columns: repeat(${numClasses + 1}, auto);">`;

    html += '<div class="cm-cell cm-header"></div>';
    classes.forEach(cls => {
        html += `<div class="cm-cell cm-header">Pred ${cls}</div>`;
    });

    for (let i = 0; i < numClasses; i++) {
        html += `<div class="cm-cell cm-header">Actual ${classes[i]}</div>`;

        for (let j = 0; j < numClasses; j++) {
            const value = matrix[i][j];
            const intensity = maxValue > 0 ? (value / maxValue) : 0;
            const color = i === j
                ? `rgba(16, 185, 129, ${0.3 + intensity * 0.7})`
                : `rgba(239, 68, 68, ${0.2 + intensity * 0.6})`;

            html += `<div class="cm-cell cm-value" style="background: ${color};" title="Actual: ${classes[i]}, Predicted: ${classes[j]}, Count: ${value}">${value}</div>`;
        }
    }

    html += '</div>';
    return html;
}

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', function () {
    console.log('🚀 Aplikasi GLCM, KNN & Naive Bayes siap digunakan!');
});
