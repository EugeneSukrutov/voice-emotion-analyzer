"""
Построение и сохранение графиков для визуализации тембра голоса.
"""
import os
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def create_plots(audio_path: str, save_dir: str) -> list:
    """
    Создаёт три графика и сохраняет их в save_dir.
    Возвращает список путей к сохранённым PNG.
    """
    os.makedirs(save_dir, exist_ok=True)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    saved_files = []

    # 1. Осциллограмма (волновая форма)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.7)
    ax.set_title("Waveform")
    ax.set_xlabel("Time, (s)")
    ax.set_ylabel("Amplitude")
    path1 = os.path.join(save_dir, "waveform.png")
    plt.savefig(path1, dpi=100, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path1)

    # 2. Тепловая карта MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax, cmap='viridis')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("MFCC Heatmap")
    path2 = os.path.join(save_dir, "mfcc_heatmap.png")
    plt.savefig(path2, dpi=100, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path2)

    # 3. Усреднённый амплитудный спектр
    D = np.abs(librosa.stft(y))
    avg_spectrum = np.mean(D, axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs, avg_spectrum)
    ax.set_title("Average Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim([0, 8000])
    path3 = os.path.join(save_dir, 'average_spectrum.png')
    plt.savefig(path3, dpi=100, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(path3)

    return saved_files
