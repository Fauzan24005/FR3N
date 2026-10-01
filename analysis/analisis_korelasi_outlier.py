"""
=============================================================================
ANALISIS KORELASI DAN DETEKSI OUTLIER PADA DATASET SENSOR INDUSTRIAL METROPT-3
Tugas Proyek UTS Data Mining - S1 Teknik Informatika
=============================================================================
Penelitian: Klasterisasi Karakteristik Operasional Sensor Kompresor Udara Metro
            Menggunakan Algoritma K-Means Berdasarkan 7 Atribut Deret Waktu Multivariat

Fitur Sensor Analog (7 Atribut Kontinu):
1. TP2             : Tekanan kompresor (bar)
2. TP3             : Tekanan panel pneumatik (bar)
3. H1              : Penurunan tekanan filter pemisah siklonik (bar)
4. DV_pressure     : Penurunan tekanan katup pengering menara udara (bar)
5. Reservoirs      : Tekanan hilir tangki penampung (bar)
6. Oil_temperature : Suhu oli kompresor (°C)
7. Motor_current   : Arus fasa motor kompresor (A)

Metode yang diterapkan sesuai acuan paper & gap analysis:
- Korelasi: Matriks Korelasi Pearson, Heatmap, Scatter Matrix, Bivariate Scatter
- Outlier : IQR (Tukey's Fences), Z-Score Standardized, Quantile Clipping 1%-99%
=============================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Pastikan output konsol mendukung karakter UTF-8 di Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Konfigurasi Tampilan & Output
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_visualisasi')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_dataset_path(custom_path=None):
    """
    Mencari lokasi file dataset MetroPT3(AirCompressor).csv secara otomatis
    pada beberapa kemungkinan struktur direktori repositori.
    """
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)
    
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return os.path.abspath(sys.argv[1])
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    cwd = os.getcwd()
    
    candidates = [
        # Struktur Repo: <repo_root>/dataset/MetroPT3(AirCompressor).csv (Script di analysis/)
        os.path.join(parent_dir, 'dataset', 'MetroPT3(AirCompressor).csv'),
        # Path spesifik pengguna
        r'd:\uts_datmin\FR3N\dataset\MetroPT3(AirCompressor).csv',
        r'd:\uts datmin\FR3N\dataset\MetroPT3(AirCompressor).csv',
        # Folder dataset relatif terhadap direktori kerja saat ini
        os.path.join(cwd, 'dataset', 'MetroPT3(AirCompressor).csv'),
        os.path.join(cwd, 'FR3N', 'dataset', 'MetroPT3(AirCompressor).csv'),
        # Folder archive (struktur lama)
        os.path.join(script_dir, 'archive', 'MetroPT3(AirCompressor).csv'),
        os.path.join(parent_dir, 'archive', 'MetroPT3(AirCompressor).csv'),
        os.path.join(grandparent_dir, 'archive', 'MetroPT3(AirCompressor).csv'),
        r'd:\uts_datmin\archive\MetroPT3(AirCompressor).csv',
        r'd:\uts datmin\archive\MetroPT3(AirCompressor).csv',
        # Langsung di folder script atau cwd
        os.path.join(script_dir, 'MetroPT3(AirCompressor).csv'),
        os.path.join(cwd, 'MetroPT3(AirCompressor).csv')
    ]
    
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
            
    # Default jika tidak ada yang ditemukan
    return os.path.abspath(os.path.join(parent_dir, 'dataset', 'MetroPT3(AirCompressor).csv'))


DATASET_PATH = find_dataset_path()

ANALOG_FEATURES = [
    'TP2', 
    'TP3', 
    'H1', 
    'DV_pressure', 
    'Reservoirs', 
    'Oil_temperature', 
    'Motor_current'
]

FEATURE_LABELS = {
    'TP2': 'TP2 (Tekanan Kompresor - bar)',
    'TP3': 'TP3 (Tekanan Pneumatik - bar)',
    'H1': 'H1 (Drop Tekanan Filter - bar)',
    'DV_pressure': 'DV_pressure (Tekanan Menara - bar)',
    'Reservoirs': 'Reservoirs (Tekanan Tangki - bar)',
    'Oil_temperature': 'Oil_temperature (Suhu Oli - °C)',
    'Motor_current': 'Motor_current (Arus Motor - A)'
}


def load_dataset(filepath=None):
    """
    Memuat dataset MetroPT-3 khusus 7 sensor analog + timestamp.
    Efisien memori dengan hanya membaca kolom yang diperlukan.
    """
    print("=" * 70)
    print("1. MEMUAT DATASET SENSOR METROPT-3")
    print("=" * 70)
    t0 = time.time()
    
    if filepath is None:
        filepath = find_dataset_path()
        
    print(f"Mencari file di: {filepath}")
    
    if not os.path.isfile(filepath):
        print(f"\n[ERROR] File dataset tidak ditemukan di: {filepath}")
        print("\nKemungkinan penyebab:")
        print("1. Lokasi file berbeda dari struktur repositori.")
        print("2. Anda dapat menentukan lokasi dataset secara langsung via command-line:")
        print("   python analisis_korelasi_outlier.py \"d:\\uts_datmin\\FR3N\\dataset\\MetroPT3(AirCompressor).csv\"")
        raise FileNotFoundError(f"Dataset tidak ditemukan di '{filepath}'")
    
    usecols = ['timestamp'] + ANALOG_FEATURES
    print(f"[OK] Membaca file: {filepath}")
    df = pd.read_csv(filepath, usecols=usecols)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    elapsed = time.time() - t0
    print(f"[OK] Selesai memuat {len(df):,} baris dan {df.shape[1]} kolom dalam {elapsed:.2f} detik.")
    print(f"Rentang Waktu: {df['timestamp'].min()} s.d. {df['timestamp'].max()}")
    print("-" * 70)
    return df


def compute_descriptive_stats(df):
    """
    Menghitung statistik deskriptif lengkap (Mean, Std, Min, Max, Skewness, Kurtosis).
    Sesuai Tabel II pada dokumen Paper UTS.
    """
    print("\n" + "=" * 70)
    print("2. STATISTIK DESKRIPTIF 7 SENSOR ANALOG")
    print("=" * 70)
    
    stats_list = []
    for col in ANALOG_FEATURES:
        series = df[col]
        stats_list.append({
            'Fitur': col,
            'Deskripsi': FEATURE_LABELS[col],
            'Mean (mu)': series.mean(),
            'Std (sigma)': series.std(),
            'Min': series.min(),
            'Median': series.median(),
            'Maks': series.max(),
            'Skewness': series.skew(),
            'Kurtosis': series.kurtosis()
        })
    
    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False))
    
    # Simpan ke CSV
    csv_path = os.path.join(OUTPUT_DIR, 'tabel_statistik_deskriptif.csv')
    stats_df.to_csv(csv_path, index=False)
    print(f"\n[Export] Tersimpan di: {csv_path}")
    print("-" * 70)
    return stats_df


def analyze_correlation(df):
    """
    Menghitung dan memvisualisasikan matriks korelasi Pearson.
    Output: Heatmap Pearson Correlation dan Tabel Korelasi CSV.
    """
    print("\n" + "=" * 70)
    print("3. ANALISIS KORELASI ANTAR-SENSOR (PEARSON)")
    print("=" * 70)
    
    corr_matrix = df[ANALOG_FEATURES].corr(method='pearson')
    print("Matriks Korelasi Pearson:")
    print(corr_matrix.round(4).to_string())
    
    # Simpan ke CSV
    csv_path = os.path.join(OUTPUT_DIR, 'tabel_korelasi_pearson.csv')
    corr_matrix.to_csv(csv_path)
    print(f"\n[Export] Tabel korelasi tersimpan di: {csv_path}")
    
    # -----------------------------------------------------------------------
    # Visualisasi 1: Heatmap Matriks Korelasi
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Full heatmap dengan anotasi angka yang jelas
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=1.2, 
        linecolor='white',
        cbar_kws={'label': 'Koefisien Korelasi Pearson (r)', 'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title('Matriks Korelasi Pearson Antar-Sensor Analog MetroPT-3', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    out_fig = os.path.join(OUTPUT_DIR, '01_matriks_korelasi_heatmap.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Heatmap tersimpan di: {out_fig}")
    
    # -----------------------------------------------------------------------
    # Visualisasi 2: Scatter Plot Pasangan Korelasi Utama (Bivariate Relationships)
    # -----------------------------------------------------------------------
    # Ambil sampel representatif 30.000 titik agar scatter plot jelas (tidak overplotting)
    sample_df = df.sample(n=min(30000, len(df)), random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    pairs = [
        ('TP2', 'H1', 'Korelasi Negatif Kuat (r = -0.961)\nTekanan Kompresor vs Drop Tekanan Filter', '#d9534f'),
        ('TP2', 'Motor_current', 'Korelasi Positif Kuat (r = +0.698)\nTekanan Kompresor vs Beban Arus Motor', '#0275d8'),
        ('TP3', 'Reservoirs', 'Korelasi Sempurna / Redundansi (r = 1.000)\nTekanan Pneumatik vs Tekanan Tangki Hilir', '#5cb85c'),
        ('Oil_temperature', 'Motor_current', 'Korelasi Positif Sedang (r = +0.529)\nSuhu Oli vs Konsumsi Arus Motor', '#f0ad4e')
    ]
    
    for ax, (x_col, y_col, title, color) in zip(axes.flatten(), pairs):
        # Scatter dengan transparansi alpha untuk memperlihatkan densitas
        ax.scatter(sample_df[x_col], sample_df[y_col], alpha=0.15, s=10, color=color, edgecolors='none', label='Observasi Sensor')
        
        # Garis tren regresi linier
        sns.regplot(
            data=sample_df, x=x_col, y=y_col, ax=ax,
            scatter=False, color='#222222', line_kws={'linewidth': 2, 'linestyle': '--', 'label': 'Garis Tren Linear'}
        )
        
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel(FEATURE_LABELS[x_col], fontsize=9)
        ax.set_ylabel(FEATURE_LABELS[y_col], fontsize=9)
        ax.grid(True)
        ax.legend(loc='best', framealpha=0.9, fontsize=8.5)
    
    plt.suptitle('Analisis Scatter Plot Bivariat Pasangan Sensor Kunci MetroPT-3', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    out_fig2 = os.path.join(OUTPUT_DIR, '02_scatter_plot_korelasi_utama.png')
    plt.savefig(out_fig2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Scatter plot korelasi utama tersimpan di: {out_fig2}")
    
    # -----------------------------------------------------------------------
    # Visualisasi 3: Scatter Plot Matrix (Pairplot Ringkas 5 Sensor Utama)
    # -----------------------------------------------------------------------
    print("Membuat Scatter Plot Matrix (Pairplot)...")
    key_features = ['TP2', 'TP3', 'H1', 'Oil_temperature', 'Motor_current']
    pair_sample = df[key_features].sample(n=min(5000, len(df)), random_state=42)
    
    g = sns.pairplot(
        pair_sample, 
        corner=True, 
        diag_kind='kde',
        plot_kws={'alpha': 0.25, 's': 8, 'color': '#1f77b4'},
        diag_kws={'fill': True, 'color': '#2ca02c'}
    )
    g.fig.suptitle('Scatter Matrix & Distribusi KDE Fitur Sensor Kunci (Data Mining)', y=1.02, fontsize=13, fontweight='bold')
    out_fig3 = os.path.join(OUTPUT_DIR, '03_pairplot_multivariat.png')
    g.savefig(out_fig3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Scatter Matrix Pairplot tersimpan di: {out_fig3}")
    print("-" * 70)


def analyze_outliers(df):
    """
    Melakukan deteksi dan analisis pencilan (outlier) dengan 2 metode ilmiah:
    1. Metode IQR (Interquartile Range / Tukey's Fences)
    2. Metode Quantile Clipping (Persentil 1% - 99% / Winsorization) sesuai Paper Rujukan
    
    Visualisasi:
    - Boxplot individual (skala asli per fitur)
    - Boxplot gabungan ternormalisasi Z-Score (memperlihatkan scale disparity & fliers)
    - Boxplot perbandingan Sebelum vs Sesudah Quantile Clipping
    - Scatter Plot deteksi outlier pada deret waktu dan bivariat
    """
    print("\n" + "=" * 70)
    print("4. ANALISIS DAN DETEKSI OUTLIER")
    print("=" * 70)
    
    # -----------------------------------------------------------------------
    # Perhitungan Outlier Metode IQR
    # -----------------------------------------------------------------------
    iqr_results = []
    outlier_masks_iqr = {}
    
    for col in ANALOG_FEATURES:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        mask = (series < lower_bound) | (series > upper_bound)
        outlier_masks_iqr[col] = mask
        count = mask.sum()
        pct = (count / len(df)) * 100
        
        iqr_results.append({
            'Fitur': col,
            'Q1 (25%)': q1,
            'Q3 (75%)': q3,
            'IQR': iqr,
            'Lower Bound (Q1-1.5*IQR)': lower_bound,
            'Upper Bound (Q3+1.5*IQR)': upper_bound,
            'Jumlah Outlier': count,
            'Persentase (%)': pct
        })
    
    iqr_df = pd.DataFrame(iqr_results)
    print("--- Ringkasan Deteksi Outlier (Metode IQR / Tukey's Fences) ---")
    print(iqr_df.to_string(index=False))
    
    csv_iqr = os.path.join(OUTPUT_DIR, 'tabel_outlier_iqr.csv')
    iqr_df.to_csv(csv_iqr, index=False)
    print(f"[Export] Tabel Outlier IQR tersimpan di: {csv_iqr}")
    
    # -----------------------------------------------------------------------
    # Perhitungan Outlier Metode Quantile (1% - 99% Winsorization - Acuan Paper)
    # -----------------------------------------------------------------------
    quant_results = []
    clipped_series_dict = {}
    
    for col in ANALOG_FEATURES:
        series = df[col]
        p1 = series.quantile(0.01)
        p99 = series.quantile(0.99)
        
        outside_mask = (series < p1) | (series > p99)
        count_outside = outside_mask.sum()
        pct_outside = (count_outside / len(df)) * 100
        
        # Winsorize / clip
        clipped_series_dict[col] = series.clip(lower=p1, upper=p99)
        
        quant_results.append({
            'Fitur': col,
            'P1 (Batas Bawah 1%)': p1,
            'P99 (Batas Atas 99%)': p99,
            'Titik Terdampak': count_outside,
            'Persentase (%)': pct_outside
        })
        
    quant_df = pd.DataFrame(quant_results)
    print("\n--- Ringkasan Quantile Clipping 1% - 99% (Sesuai Metodologi Paper) ---")
    print(quant_df.to_string(index=False))
    
    csv_quant = os.path.join(OUTPUT_DIR, 'tabel_outlier_quantile_clipping.csv')
    quant_df.to_csv(csv_quant, index=False)
    print(f"[Export] Tabel Quantile Clipping tersimpan di: {csv_quant}")
    
    # -----------------------------------------------------------------------
    # Visualisasi 4: Box Plot Individual Multi-Panel (Skala Asli Per Fitur)
    # -----------------------------------------------------------------------
    # Gunakan sampel 50.000 titik untuk rendering boxplot cepat dan bersih
    sample_df = df[ANALOG_FEATURES].sample(n=min(50000, len(df)), random_state=42)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd', '#8c564b']
    
    for i, col in enumerate(ANALOG_FEATURES):
        ax = axes[i]
        sns.boxplot(
            y=sample_df[col], 
            ax=ax, 
            color=colors[i],
            width=0.4,
            flierprops=dict(marker='o', markersize=3, alpha=0.3, markerfacecolor='red', markeredgecolor='none')
        )
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_ylabel(FEATURE_LABELS[col], fontsize=9)
        ax.grid(True)
    
    # Nonaktifkan subplot ke-8 (karena hanya ada 7 fitur)
    axes[7].axis('off')
    
    # Tambahkan kotak penjelasan metrik pada slot kosong
    text_info = (
        "Interpretasi Box Plot:\n"
        "• Garis Tengah: Median (Q2)\n"
        "• Kotak: Rentang Interkuartil (IQR = Q3 - Q1)\n"
        "• Whiskers: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]\n"
        "• Titik Merah: Pencilan (Outliers / Fliers)\n\n"
        "Catatan Khusus:\n"
        "DV_pressure memiliki kurtosis 38.7\n"
        "dengan rentang fliers terjauh akibat\n"
        "lonjakan transien siklus pengeringan."
    )
    axes[7].text(0.1, 0.45, text_info, fontsize=9.5, family='sans-serif',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#cccccc'))
    
    plt.suptitle('Visualisasi Box Plot Sebaran & Outlier Per Sensor Analog (Skala Asli)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_fig4 = os.path.join(OUTPUT_DIR, '04_boxplot_individual_skala_asli.png')
    plt.savefig(out_fig4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Visualisasi] Box plot individual tersimpan di: {out_fig4}")
    
    # -----------------------------------------------------------------------
    # Visualisasi 5: Box Plot Standardized Z-Score (Menunjukkan Ketimpangan Skala)
    # -----------------------------------------------------------------------
    # Normalisasi Z-Score: (x - mean) / std
    zscore_df = (sample_df - sample_df.mean()) / sample_df.std()
    
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.boxplot(
        data=zscore_df, 
        ax=ax, 
        palette='Set2', 
        width=0.5,
        flierprops=dict(marker='o', markersize=3, alpha=0.35, markerfacecolor='#e74c3c', markeredgecolor='none')
    )
    
    # Garis ambang batas umum 3-sigma (|Z| = 3)
    ax.axhline(3, color='red', linestyle='--', linewidth=1.2, label='Batas Atas 3-Sigma (+3σ)')
    ax.axhline(-3, color='red', linestyle='--', linewidth=1.2, label='Batas Bawah 3-Sigma (-3σ)')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    ax.set_title('Distribusi Data Ternormalisasi Z-Score & Deteksi Outlier Ekstrem (> 3σ)', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Standar Deviasi dari Rata-rata (Z-Score)', fontsize=10)
    ax.set_xlabel('Sensor Analog', fontsize=10)
    ax.grid(True)
    ax.legend(loc='upper right', framealpha=0.95)
    plt.xticks(rotation=15, fontsize=9.5)
    plt.tight_layout()
    
    out_fig5 = os.path.join(OUTPUT_DIR, '05_boxplot_standardized_zscore.png')
    plt.savefig(out_fig5, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Box plot standardized tersimpan di: {out_fig5}")
    
    # -----------------------------------------------------------------------
    # Visualisasi 6: Perbandingan Box Plot Sebelum vs Sesudah Quantile Clipping (Winsorization)
    # -----------------------------------------------------------------------
    # Sesuai metode yang dipilih pada paper rujukan kelompok
    clipped_sample_df = pd.DataFrame({col: clipped_series_dict[col].iloc[sample_df.index] for col in ANALOG_FEATURES})
    
    # Gabungkan dalam format perbandingan untuk 4 sensor paling fluktuatif
    compare_features = ['TP2', 'H1', 'DV_pressure', 'Motor_current']
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    
    for i, col in enumerate(compare_features):
        # Sebelum clipping
        ax_before = axes[0, i]
        sns.boxplot(
            y=sample_df[col], ax=ax_before, color='#e74c3c', width=0.4,
            flierprops=dict(marker='o', markersize=3, alpha=0.4, markerfacecolor='black', markeredgecolor='none')
        )
        ax_before.set_title(f'Sebelum: {col}', fontsize=10.5, fontweight='bold', color='#c0392b')
        ax_before.set_ylabel('Nilai Mentah', fontsize=8.5)
        ax_before.grid(True)
        
        # Sesudah clipping
        ax_after = axes[1, i]
        sns.boxplot(
            y=clipped_sample_df[col], ax=ax_after, color='#2ecc71', width=0.4,
            flierprops=dict(marker='o', markersize=3, alpha=0.4, markerfacecolor='black', markeredgecolor='none')
        )
        ax_after.set_title(f'Sesudah Winsorize: {col}', fontsize=10.5, fontweight='bold', color='#27ae60')
        ax_after.set_ylabel('Nilai Ter-clipping (1%-99%)', fontsize=8.5)
        ax_after.grid(True)
        
    plt.suptitle('Efektivitas Penanganan Outlier: Sebelum vs Sesudah Quantile Clipping (1% - 99%)', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    out_fig6 = os.path.join(OUTPUT_DIR, '06_boxplot_before_after_clipping.png')
    plt.savefig(out_fig6, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Box plot before-after clipping tersimpan di: {out_fig6}")
    
    # -----------------------------------------------------------------------
    # Visualisasi 7: Scatter Plot Bivariat dengan Deteksi Outlier Teranotasi
    # -----------------------------------------------------------------------
    # Memvisualisasikan di mana titik outlier berada pada ruang 2D sensor kunci
    scatter_sample = df.sample(n=min(25000, len(df)), random_state=42).copy()
    
    # Deteksi outlier berbasis Quantile Clipping 1% - 99% (Sesuai Paper)
    p1_tp2 = quant_df.loc[quant_df['Fitur'] == 'TP2', 'P1 (Batas Bawah 1%)'].values[0]
    p99_tp2 = quant_df.loc[quant_df['Fitur'] == 'TP2', 'P99 (Batas Atas 99%)'].values[0]
    p1_mc = quant_df.loc[quant_df['Fitur'] == 'Motor_current', 'P1 (Batas Bawah 1%)'].values[0]
    p99_mc = quant_df.loc[quant_df['Fitur'] == 'Motor_current', 'P99 (Batas Atas 99%)'].values[0]
    p1_oil = quant_df.loc[quant_df['Fitur'] == 'Oil_temperature', 'P1 (Batas Bawah 1%)'].values[0]
    p99_oil = quant_df.loc[quant_df['Fitur'] == 'Oil_temperature', 'P99 (Batas Atas 99%)'].values[0]

    outlier_p1 = (scatter_sample['TP2'] < p1_tp2) | (scatter_sample['TP2'] > p99_tp2) | \
                 (scatter_sample['Motor_current'] < p1_mc) | (scatter_sample['Motor_current'] > p99_mc)
    
    outlier_p2 = (scatter_sample['Oil_temperature'] < p1_oil) | (scatter_sample['Oil_temperature'] > p99_oil) | \
                 (scatter_sample['Motor_current'] < p1_mc) | (scatter_sample['Motor_current'] > p99_mc)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    
    # Panel 1: TP2 vs Motor_current
    ax1.scatter(scatter_sample.loc[~outlier_p1, 'TP2'], scatter_sample.loc[~outlier_p1, 'Motor_current'],
                color='#3498db', alpha=0.25, s=15, label='Kondisi Normal (Inlier)')
    ax1.scatter(scatter_sample.loc[outlier_p1, 'TP2'], scatter_sample.loc[outlier_p1, 'Motor_current'],
                color='#e74c3c', alpha=0.85, s=35, edgecolors='black', linewidths=0.5, label='Pencilan Terdeteksi (< P1 atau > P99)')
    ax1.set_title('Tekanan Kompresor (TP2) vs Arus Motor (Motor_current)', fontsize=11.5, fontweight='bold')
    ax1.set_xlabel(FEATURE_LABELS['TP2'], fontsize=9.5)
    ax1.set_ylabel(FEATURE_LABELS['Motor_current'], fontsize=9.5)
    ax1.grid(True)
    ax1.legend(loc='upper left', framealpha=0.95)
    
    # Panel 2: Oil_temperature vs Motor_current
    ax2.scatter(scatter_sample.loc[~outlier_p2, 'Oil_temperature'], scatter_sample.loc[~outlier_p2, 'Motor_current'],
                color='#2ecc71', alpha=0.25, s=15, label='Kondisi Normal (Inlier)')
    ax2.scatter(scatter_sample.loc[outlier_p2, 'Oil_temperature'], scatter_sample.loc[outlier_p2, 'Motor_current'],
                color='#e67e22', alpha=0.85, s=35, edgecolors='black', linewidths=0.5, label='Pencilan Suhu/Beban (< P1 atau > P99)')
    ax2.set_title('Suhu Oli (Oil_temperature) vs Arus Motor (Motor_current)', fontsize=11.5, fontweight='bold')
    ax2.set_xlabel(FEATURE_LABELS['Oil_temperature'], fontsize=9.5)
    ax2.set_ylabel(FEATURE_LABELS['Motor_current'], fontsize=9.5)
    ax2.grid(True)
    ax2.legend(loc='upper left', framealpha=0.95)
    
    plt.suptitle('Analisis Scatter Plot Bivariat dengan Deteksi Outlier Ekstrem (Kuantil 1% - 99%)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    out_fig7 = os.path.join(OUTPUT_DIR, '07_scatter_outlier_bivariat.png')
    plt.savefig(out_fig7, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Scatter plot outlier bivariat tersimpan di: {out_fig7}")
    
    # -----------------------------------------------------------------------
    # Visualisasi 8: Scatter Plot Deret Waktu Spike Pencilan (Transient Spikes)
    # -----------------------------------------------------------------------
    # Mengambil sekuens waktu 2 minggu (misal 15.000 titik berurutan) untuk melihat lonjakan transien
    time_subset = df.iloc[50000:65000].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: DV_pressure spikes
    p99_dv = quant_df.loc[quant_df['Fitur'] == 'DV_pressure', 'P99 (Batas Atas 99%)'].values[0]
    outliers_dv = time_subset[time_subset['DV_pressure'] > p99_dv]
    
    ax1.plot(time_subset['timestamp'], time_subset['DV_pressure'], color='#34495e', linewidth=0.8, alpha=0.7, label='Sinyal Kontinu DV_pressure')
    ax1.scatter(outliers_dv['timestamp'], outliers_dv['DV_pressure'], color='#e74c3c', s=20, label=f'Transient Spikes (> P99={p99_dv:.2f} bar)', zorder=5)
    ax1.axhline(p99_dv, color='#e74c3c', linestyle='--', linewidth=1.2, label='Batas P99 (Quantile 99%)')
    ax1.set_title('Deteksi Transient Spikes pada Tekanan Menara Udara (DV_pressure) dalam Deret Waktu', fontsize=11, fontweight='bold')
    ax1.set_ylabel('DV_pressure (bar)', fontsize=9.5)
    ax1.grid(True)
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Plot 2: Oil_temperature vs Batas Outlier
    p1_oil = quant_df.loc[quant_df['Fitur'] == 'Oil_temperature', 'P1 (Batas Bawah 1%)'].values[0]
    p99_oil = quant_df.loc[quant_df['Fitur'] == 'Oil_temperature', 'P99 (Batas Atas 99%)'].values[0]
    outliers_oil = time_subset[(time_subset['Oil_temperature'] < p1_oil) | (time_subset['Oil_temperature'] > p99_oil)]
    
    ax2.plot(time_subset['timestamp'], time_subset['Oil_temperature'], color='#2980b9', linewidth=0.8, alpha=0.7, label='Sinyal Kontinu Suhu Oli')
    ax2.scatter(outliers_oil['timestamp'], outliers_oil['Oil_temperature'], color='#f39c12', s=25, label='Pencilan Suhu (< P1 atau > P99)', zorder=5)
    ax2.axhline(p99_oil, color='#e67e22', linestyle='--', linewidth=1.2, label=f'Batas P99 ({p99_oil:.1f} °C)')
    ax2.axhline(p1_oil, color='#3498db', linestyle='--', linewidth=1.2, label=f'Batas P1 ({p1_oil:.1f} °C)')
    ax2.set_title('Dinamika Suhu Oli (Oil_temperature) Terhadap Batas Ambang Kuantil', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Suhu Oli (°C)', fontsize=9.5)
    ax2.set_xlabel('Waktu (Timestamp)', fontsize=9.5)
    ax2.grid(True)
    ax2.legend(loc='upper right', framealpha=0.9)
    
    plt.suptitle('Visualisasi Scatter Deret Waktu: Deteksi Pencilan Transien Fisik Kompresor', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    out_fig8 = os.path.join(OUTPUT_DIR, '08_scatter_timeseries_outlier_spikes.png')
    plt.savefig(out_fig8, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Visualisasi] Scatter plot deret waktu spikes tersimpan di: {out_fig8}")
    print("-" * 70)


def main():
    start_time = time.time()
    print("=" * 70)
    print("PROGRAM ANALISIS KORELASI DAN DETEKSI OUTLIER (DATA MINING)")
    print("Dataset: MetroPT-3 Air Compressor Multivariate Time-Series")
    print("=" * 70)
    
    # 1. Muat dataset
    df = load_dataset()
    
    # 2. Statistik Deskriptif
    compute_descriptive_stats(df)
    
    # 3. Analisis Korelasi & Scatter Plot
    analyze_correlation(df)
    
    # 4. Analisis Outlier & Box Plot
    analyze_outliers(df)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"SEMUA PROSES ANALISIS & VISUALISASI BERHASIL DISELESAIKAN!")
    print(f"Total Waktu Eksekusi: {total_time:.2f} detik")
    print(f"Seluruh output tersimpan di direktori: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
