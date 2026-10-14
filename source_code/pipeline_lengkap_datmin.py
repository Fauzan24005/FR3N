"""
===============================================================================
PIPELINE LENGKAP KLASTERISASI METROPT-3 AIR COMPRESSOR
Penelitian: Klasterisasi Karakteristik Operasional Sensor Kompresor Udara Metro
            Menggunakan Algoritma K-Means Berdasarkan 7 Atribut Deret Waktu Multivariat

Pipeline:
1. Load Data + Missing Values Check
2. Statistik Deskriptif
3. Analisis Korelasi Pearson
4. Deteksi Outlier (IQR + Quantile Clipping)
5. Preprocessing: EWMA + Quantile Clipping + Sliding Window
6. Ekstraksi Fitur: Domain Waktu + Domain Frekuensi (FFT)
7. Standardisasi: StandardScaler
8. Reduksi Dimensi: PCA (95% variance)
9. Grid Search Hiperparameter K-Means
10. K-Means Final dengan H* Optimal
11. Visualisasi Hasil Klaster (PC1-PC2, Radar, Distribusi)
12. Evaluasi Metrik (Silhouette, DBI, CHI)
===============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import product
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings('ignore')

# ==========================================
# KONFIGURASI GLOBAL
# ==========================================
CSV_PATH = "MetroPT3(AirCompressor).csv"
OUTPUT_DIR = "hasil_pipeline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KOLOM = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
FEATURE_LABELS = {
    'TP2': 'TP2 (Tekanan Kompresor - bar)',
    'TP3': 'TP3 (Tekanan Pneumatik - bar)',
    'H1': 'H1 (Drop Tekanan Filter - bar)',
    'DV_pressure': 'DV_pressure (Tekanan Menara - bar)',
    'Reservoirs': 'Reservoirs (Tekanan Tangki - bar)',
    'Oil_temperature': 'Oil_temperature (Suhu Oli - °C)',
    'Motor_current': 'Motor_current (Arus Motor - A)'
}

# Parameter Preprocessing
WINDOW_SIZE = 5
STEP_SIZE = 1
EWMA_SPAN = 10
Q_LOW, Q_HIGH = 0.01, 0.99
N_COMPONENTS = 0.95  # Target variansi PCA

# Parameter Grid Search K-Means
PARAM_GRID = {
    "n_clusters": range(2, 11),
    "init": ["k-means++", "random"],
    "n_init": [10, 20],
    "max_iter": [300, 500],
    "tol": [1e-4, 1e-3],
}
SAMPLE_SIZE_GRIDSEARCH = 200000
EVAL_SIZE_GRIDSEARCH = 10000

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def simpan_fig(nama):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, nama), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f'[Gambar] {nama}')

print("=" * 80)
print("PIPELINE LENGKAP KLASTERISASI METROPT-3")
print("=" * 80)
start_total = time.time()

# ==========================================
# 1. LOAD DATA
# ==========================================
print("\n[1/12] Load Dataset...")
t0 = time.time()
df_raw = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
df = df_raw[['timestamp'] + KOLOM].copy()
df.set_index('timestamp', inplace=True)
print(f"Dataset: {len(df):,} observasi, {len(KOLOM)} fitur sensor")
print(f"Rentang waktu: {df.index.min()} s.d. {df.index.max()}")
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 2. STATISTIK DESKRIPTIF
# ==========================================
print("\n[2/12] Statistik Deskriptif...")
t0 = time.time()
stats_list = []
for col in KOLOM:
    s = df[col]
    stats_list.append({
        'Fitur': col,
        'Mean': s.mean(),
        'Std': s.std(),
        'Min': s.min(),
        'Median': s.median(),
        'Max': s.max(),
        'Skewness': s.skew(),
        'Kurtosis': s.kurtosis()
    })
stats_df = pd.DataFrame(stats_list)
stats_df.to_csv(os.path.join(OUTPUT_DIR, 'tabel_statistik_deskriptif.csv'), index=False)
print(stats_df.to_string(index=False))
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 3. ANALISIS KORELASI PEARSON
# ==========================================
print("\n[3/12] Analisis Korelasi Pearson...")
t0 = time.time()
corr_matrix = df[KOLOM].corr(method='pearson')
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'matriks_korelasi.csv'))
print("Matriks Korelasi:")
print(corr_matrix.round(3).to_string())

# Visualisasi heatmap
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1, 
            center=0, square=True, linewidths=1.2, linecolor='white', ax=ax,
            cbar_kws={'label': 'Koefisien Korelasi Pearson'})
ax.set_title('Matriks Korelasi Pearson Sensor MetroPT-3', fontsize=13, fontweight='bold')
plt.xticks(rotation=30, ha='right')
simpan_fig('matriks_korelasi.png')
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 4. DETEKSI OUTLIER
# ==========================================
print("\n[4/12] Deteksi Outlier (Quantile Clipping 1%-99%)...")
t0 = time.time()
quant_results = []
for col in KOLOM:
    s = df[col]
    p1 = s.quantile(0.01)
    p99 = s.quantile(0.99)
    outside = ((s < p1) | (s > p99)).sum()
    pct = (outside / len(df)) * 100
    quant_results.append({
        'Fitur': col,
        'P1 (1%)': p1,
        'P99 (99%)': p99,
        'Titik Outlier': outside,
        'Persentase': f"{pct:.2f}%"
    })
quant_df = pd.DataFrame(quant_results)
quant_df.to_csv(os.path.join(OUTPUT_DIR, 'tabel_outlier_quantile.csv'), index=False)
print(quant_df.to_string(index=False))

# Visualisasi missing values
missing_awal = df.isna().sum()
plt.figure(figsize=(8, 4))
missing_awal.plot(kind='bar', color='tomato')
plt.title("Missing Values per Fitur")
plt.ylabel("Jumlah")
simpan_fig('missing_values.png')

# Boxplot outlier handling
df_before = df.copy()
lower, upper = df.quantile(Q_LOW), df.quantile(Q_HIGH)
df_clipped = df.clip(lower=lower, upper=upper, axis=1)

def minmax_norm(d): return (d - d.min()) / (d.max() - d.min() + 1e-12)
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
sns.boxplot(data=minmax_norm(df_before), ax=ax[0])
ax[0].set_title("Sebelum Outlier Handling")
ax[0].tick_params(axis='x', rotation=45)
sns.boxplot(data=minmax_norm(df_clipped), ax=ax[1])
ax[1].set_title("Sesudah Quantile Clipping (1%-99%)")
ax[1].tick_params(axis='x', rotation=45)
simpan_fig('outlier_handling.png')
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 5. PREPROCESSING: CLEANING
# ==========================================
print("\n[5/12] Preprocessing: EWMA + Quantile Clipping...")
t0 = time.time()
df_clean = df.interpolate(method='linear', limit_direction='both').bfill().ffill()
df_clean = df_clean.ewm(span=EWMA_SPAN, adjust=False).mean()
lower, upper = df_clean.quantile(Q_LOW), df_clean.quantile(Q_HIGH)
df_clean = df_clean.clip(lower=lower, upper=upper, axis=1)
print(f"Data bersih: {len(df_clean):,} observasi")
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 6. SLIDING WINDOW
# ==========================================
print("\n[6/12] Sliding Window...")
t0 = time.time()
arr = df_clean.values
idx = df_clean.index
n_win = (len(arr) - WINDOW_SIZE) // STEP_SIZE + 1
seg = sliding_window_view(arr, WINDOW_SIZE, axis=0)[::STEP_SIZE]
print(f"Jumlah jendela: {n_win:,} (window={WINDOW_SIZE}, step={STEP_SIZE})")

# Visualisasi sliding window
sig = df_clean[KOLOM[0]].iloc[:60].values
plt.figure(figsize=(11, 4))
plt.plot(sig, label=KOLOM[0], color='steelblue')
for i, c in zip(range(0, 3 * WINDOW_SIZE, WINDOW_SIZE), ['#ffb3b3', '#b3ffb3', '#b3b3ff']):
    plt.axvspan(i, i + WINDOW_SIZE - 1, color=c, alpha=0.5)
plt.title(f"Ilustrasi Sliding Window (size={WINDOW_SIZE}, step={STEP_SIZE})")
plt.xlabel("Indeks sampel")
plt.legend()
simpan_fig('sliding_window.png')
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 7. EKSTRAKSI FITUR
# ==========================================
print("\n[7/12] Ekstraksi Fitur (Domain Waktu + FFT)...")
t0 = time.time()
feat, names = [], []

# Domain waktu
for nm, fn in [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]:
    feat.append(fn(seg, axis=2))
    names += [f"{c}_{nm}" for c in KOLOM]

# Domain frekuensi (FFT)
centered = seg - seg.mean(axis=2, keepdims=True)
mag = np.abs(np.fft.rfft(centered, axis=2))
feat.append((mag ** 2).sum(axis=2))
names += [f"{c}_fft_energy" for c in KOLOM]
feat.append(mag[:, :, 1:].argmax(axis=2) + 1 if mag.shape[2] > 1 else np.zeros(seg.shape[:2]))
names += [f"{c}_fft_domfreq" for c in KOLOM]

df_feat = pd.DataFrame(np.hstack(feat), columns=names,
                       index=idx[WINDOW_SIZE - 1::STEP_SIZE][:n_win])
print(f"Fitur hasil ekstraksi: {df_feat.shape[1]} fitur dari {len(KOLOM)} sensor")

# Visualisasi FFT
w = np.fft.rfft(sig[:WINDOW_SIZE] - sig[:WINDOW_SIZE].mean())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(sig[:WINDOW_SIZE], marker='o')
ax[0].set_title("Domain Waktu (1 jendela)")
ax[1].stem(np.abs(w))
ax[1].set_title("Domain Frekuensi (FFT)")
simpan_fig('fft_time_domain.png')
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 8. STANDARDISASI
# ==========================================
print("\n[8/12] Standardisasi (StandardScaler)...")
t0 = time.time()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feat)
df_scaled = pd.DataFrame(X_scaled, columns=df_feat.columns, index=df_feat.index)

# Visualisasi StandardScaler
pilih = df_feat.columns[:8]
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=df_feat[pilih], ax=ax[0])
ax[0].set_title("Sebelum StandardScaler")
ax[0].tick_params(axis='x', rotation=45)
sns.boxplot(data=df_scaled[pilih], ax=ax[1])
ax[1].set_title("Sesudah StandardScaler (mean~0, std=1)")
ax[1].tick_params(axis='x', rotation=45)
simpan_fig('standardscaler.png')
print(f"Rentang fitur sebelum: [{df_feat.min().min():.2f}, {df_feat.max().max():.2f}]")
print(f"Rentang fitur sesudah: [{df_scaled.min().min():.2f}, {df_scaled.max().max():.2f}]")
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 9. REDUKSI DIMENSI (PCA)
# ==========================================
print("\n[9/12] Reduksi Dimensi PCA...")
t0 = time.time()
pca = PCA(n_components=N_COMPONENTS, random_state=42)
Z = pca.fit_transform(X_scaled)
d = Z.shape[1]
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
print(f"Dimensi: {X_scaled.shape[1]} -> {d} komponen (variansi {cum[-1]*100:.2f}%)")
print(f"PC1: {evr[0]*100:.2f}%, PC2: {evr[1]*100:.2f}%")

# Visualisasi PCA variance
plt.figure(figsize=(8, 4))
plt.bar(range(1, d + 1), evr, alpha=0.6, label='Per komponen')
plt.step(range(1, d + 1), cum, where='mid', color='red', label='Kumulatif')
plt.xlabel("Komponen Utama")
plt.ylabel("Rasio Variansi")
plt.legend()
plt.title("PCA - Explained Variance")
simpan_fig('pca_variance.png')

# Latent space PC1-PC2
rs = np.random.RandomState(42)
s_idx = rs.choice(len(Z), min(100000, len(Z)), replace=False)
plt.figure(figsize=(7, 6))
plt.scatter(Z[s_idx, 0], Z[s_idx, 1], s=2, alpha=0.3, c='steelblue')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Latent Space (PC1 vs PC2) - Sebelum Klasterisasi")
simpan_fig('latent_space.png')

df_z = pd.DataFrame(Z, index=df_feat.index, columns=[f"PC{i+1}" for i in range(d)])
df_z.to_csv(os.path.join(OUTPUT_DIR, 'z_latent.csv'))
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 10. GRID SEARCH HIPERPARAMETER K-MEANS
# ==========================================
print("\n[10/12] Grid Search Hiperparameter K-Means...")
print(f"Sampel training: {min(SAMPLE_SIZE_GRIDSEARCH, len(Z)):,}")
t0 = time.time()

# Sampel untuk Grid Search
Z_sample_idx = rs.choice(len(Z), min(SAMPLE_SIZE_GRIDSEARCH, len(Z)), replace=False)
Z_sample = Z[Z_sample_idx]

# Sampel evaluasi
eval_idx = rs.choice(len(Z_sample), min(EVAL_SIZE_GRIDSEARCH, len(Z_sample)), replace=False)

kombinasi = list(product(*PARAM_GRID.values()))
hasil = []

print(f"Menjalankan Grid Search: {len(kombinasi)} kombinasi...")
for nilai in tqdm(kombinasi, desc="Grid Search"):
    p = dict(zip(PARAM_GRID.keys(), nilai))
    model = KMeans(**p, random_state=42)
    label = model.fit_predict(Z_sample)
    
    hasil.append({
        **p,
        "inertia": model.inertia_,
        "silhouette": silhouette_score(Z_sample[eval_idx], label[eval_idx]),
        "davies_bouldin": davies_bouldin_score(Z_sample[eval_idx], label[eval_idx]),
        "calinski_harabasz": calinski_harabasz_score(Z_sample[eval_idx], label[eval_idx]),
    })

df_hasil = pd.DataFrame(hasil)
df_hasil.to_csv(os.path.join(OUTPUT_DIR, 'hasil_grid_search.csv'), index=False)

# Ekstraksi H* optimal (berdasarkan Silhouette tertinggi)
terbaik = df_hasil.sort_values("silhouette", ascending=False).iloc[0]
print("\n=== KONFIGURASI H* OPTIMAL (Berdasarkan Silhouette) ===")
print(terbaik)

param_terbaik = {
    "n_clusters": int(terbaik["n_clusters"]),
    "init": terbaik["init"],
    "n_init": int(terbaik["n_init"]),
    "max_iter": int(terbaik["max_iter"]),
    "tol": float(terbaik["tol"]),
}

# Tabel H* untuk paper
tabel_h = pd.DataFrame([{
    'Hiperparameter': 'K (n_clusters)',
    'Nilai Optimal': int(terbaik["n_clusters"]),
    'Search Space': '2–10'
}, {
    'Hiperparameter': 'init',
    'Nilai Optimal': terbaik["init"],
    'Search Space': 'k-means++, random'
}, {
    'Hiperparameter': 'n_init',
    'Nilai Optimal': int(terbaik["n_init"]),
    'Search Space': '10, 20'
}, {
    'Hiperparameter': 'max_iter',
    'Nilai Optimal': int(terbaik["max_iter"]),
    'Search Space': '300, 500'
}, {
    'Hiperparameter': 'tol',
    'Nilai Optimal': f"{float(terbaik['tol']):.0e}",
    'Search Space': '1e-4, 1e-3'
}, {
    'Hiperparameter': 'Silhouette (Grid)',
    'Nilai Optimal': f"{terbaik['silhouette']:.4f}",
    'Search Space': '—'
}])
tabel_h.to_csv(os.path.join(OUTPUT_DIR, 'tabel_hiperparameter_optimal.csv'), index=False)
print("\nTabel Hiperparameter H* Optimal:")
print(tabel_h.to_string(index=False))
print(f"Waktu Grid Search: {time.time()-t0:.2f}s")

# ==========================================
# 11. K-MEANS FINAL (FULL DATA)
# ==========================================
print("\n[11/12] K-Means Final (Full Data dengan H*)...")
t0 = time.time()
model_final = KMeans(**param_terbaik, random_state=42)
labels_final = model_final.fit_predict(Z)

# Evaluasi metrik final
sil_final = silhouette_score(Z[eval_idx], labels_final[eval_idx])
dbi_final = davies_bouldin_score(Z, labels_final)
chi_final = calinski_harabasz_score(Z, labels_final)

print(f"Klaster terbentuk: K = {param_terbaik['n_clusters']}")
print(f"Silhouette Score: {sil_final:.4f}")
print(f"Davies-Bouldin Index: {dbi_final:.4f}")
print(f"Calinski-Harabasz Index: {chi_final:.4f}")

# Simpan hasil
df_z['klaster'] = labels_final
df_z.to_csv(os.path.join(OUTPUT_DIR, 'hasil_klaster_final.csv'))

# Tabel metrik evaluasi
tabel_metrik = pd.DataFrame([{
    'Metrik': 'Silhouette Score (Grid Search)',
    'Nilai': f"{terbaik['silhouette']:.4f}",
    'Interpretasi': 'Kohesi & separasi klaster (Grid)'
}, {
    'Metrik': 'Silhouette Score (Final)',
    'Nilai': f"{sil_final:.4f}",
    'Interpretasi': 'Kohesi & separasi klaster (Full data)'
}, {
    'Metrik': 'Davies-Bouldin Index',
    'Nilai': f"{dbi_final:.4f}",
    'Interpretasi': 'Rasio similaritas intra/inter (lebih rendah lebih baik)'
}, {
    'Metrik': 'Calinski-Harabasz Index',
    'Nilai': f"{chi_final:.2f}",
    'Interpretasi': 'Variance ratio (lebih tinggi lebih baik)'
}])
tabel_metrik.to_csv(os.path.join(OUTPUT_DIR, 'tabel_metrik_evaluasi.csv'), index=False)
print("\nTabel Metrik Evaluasi:")
print(tabel_metrik.to_string(index=False))
print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# 12. VISUALISASI HASIL KLASTER
# ==========================================
print("\n[12/12] Visualisasi Hasil Klaster...")
t0 = time.time()

# 12a. Scatter PC1-PC2 dengan warna klaster
plt.figure(figsize=(8, 6))
for k in range(param_terbaik['n_clusters']):
    mask = labels_final == k
    plt.scatter(Z[mask, 0], Z[mask, 1], s=3, alpha=0.4, label=f'Klaster {k}')
# Centroid
plt.scatter(model_final.cluster_centers_[:, 0], model_final.cluster_centers_[:, 1],
            s=200, c='red', marker='X', edgecolors='black', linewidths=2, label='Centroid')
plt.xlabel(f"PC1 ({evr[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({evr[1]*100:.2f}%)")
plt.title(f"Hasil Klasterisasi K-Means (K={param_terbaik['n_clusters']}) pada Latent Space PCA")
plt.legend()
simpan_fig('hasil_klaster_pca.png')

# 12b. Distribusi klaster
klaster_dist = pd.Series(labels_final).value_counts().sort_index()
plt.figure(figsize=(6, 4))
klaster_dist.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title("Distribusi Jumlah Observasi per Klaster")
plt.xlabel("Klaster")
plt.ylabel("Jumlah Observasi")
plt.xticks(rotation=0)
for i, v in enumerate(klaster_dist):
    plt.text(i, v + len(Z)*0.01, f"{v:,}", ha='center', fontsize=9)
simpan_fig('distribusi_klaster.png')
print("Distribusi klaster:")
print(klaster_dist)

# 12c. Radar chart karakteristik klaster (centroid di fitur asli)
# Project centroid kembali ke space fitur asli (inverse transform)
centroid_latent = model_final.cluster_centers_  # (K, d)
centroid_scaled = pca.inverse_transform(centroid_latent)  # (K, n_feat)
centroid_original = scaler.inverse_transform(centroid_scaled)  # (K, n_feat)

# Ambil fitur mean dari 7 sensor (fitur 0-6 adalah TP2_mean..Motor_current_mean)
mean_features_idx = [i for i, name in enumerate(names) if '_mean' in name][:7]
centroid_means = centroid_scaled[:, mean_features_idx]  # (K, 7)

# Normalize ke [0,1] untuk radar chart
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
centroid_norm = mm_scaler.fit_transform(centroid_means.T).T  # (K, 7)

angles = np.linspace(0, 2 * np.pi, len(KOLOM), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
for k in range(param_terbaik['n_clusters']):
    values = centroid_norm[k].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Klaster {k}')
    ax.fill(angles, values, alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(KOLOM)
ax.set_ylim(0, 1)
ax.set_title(f"Profil Karakteristik Sensor per Klaster (K={param_terbaik['n_clusters']})", 
             fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
simpan_fig('radar_chart.png')

# 12d. Time series plot klaster (sample)
# Labels match windowed data (len = 1,516,944), original df_clean = 1,516,948
# Window (window=5, step=1) keeps last 1,516,944 rows
df_clean_labeled = df_clean.iloc[-len(labels_final):].copy()
df_clean_labeled['klaster'] = labels_final
sample_time = df_clean_labeled.iloc[:5000]

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for k in range(min(3, param_terbaik['n_clusters'])):
    mask = sample_time['klaster'] == k
    axes[k].plot(sample_time.index[mask], sample_time.loc[mask, 'TP2'], 
                 'o', markersize=2, alpha=0.6, label=f'Klaster {k}')
    axes[k].set_ylabel('TP2 (bar)')
    axes[k].set_title(f'Klaster {k}')
    axes[k].legend()
    axes[k].grid(True)
axes[-1].set_xlabel('Timestamp')
plt.suptitle('Time Series TP2 per Klaster (Sample 5000 observasi)', fontsize=13, fontweight='bold')
simpan_fig('time_series_plot.png')

print(f"Waktu: {time.time()-t0:.2f}s")

# ==========================================
# RINGKASAN AKHIR
# ==========================================
elapsed_total = time.time() - start_total
print("\n" + "=" * 80)
print("PIPELINE SELESAI!")
print("=" * 80)
print(f"Total waktu eksekusi: {elapsed_total:.2f} detik ({elapsed_total/60:.2f} menit)")
print(f"\nHasil tersimpan di: {OUTPUT_DIR}/")
print("\nFile output:")
print("  - tabel_statistik_deskriptif.csv")
print("  - matriks_korelasi.csv + matriks_korelasi.png")
print("  - tabel_outlier_quantile.csv")
print("  - missing_values.png, outlier_handling.png")
print("  - sliding_window.png, fft_time_domain.png")
print("  - standardscaler.png, pca_variance.png, latent_space.png")
print("  - z_latent.csv")
print("  - hasil_grid_search.csv, tabel_hiperparameter_optimal.csv")
print("  - hasil_klaster_final.csv, tabel_metrik_evaluasi.csv")
print("  - hasil_klaster_pca.png, distribusi_klaster.png, radar_chart.png")
print("  - time_series_plot.png")
print("\n" + "=" * 80)
