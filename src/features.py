"""
Извлечение акустических признаков из аудиофайла.
"""
import librosa
import numpy as np


def extract_features(file_path: str, sr: int = 16000, n_mfcc: int = 40) -> dict:
    """
    Загружает аудиофайл и возвращает словарь признаков:
    - mfcc_mean: усреднённые по времени MFCC (массив длины n_mfcc)
    - spectral_centroid_mean: средний спектральный центроид
    - spectral_rolloff_mean: средний спад спектра
    - tempo: оценка темпа (BPM)
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        raise ValueError(f"Не удалось загрузить аудио: {e}")

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Спектральный центроид (яркость)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    # Спектральный спад (roll-off)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = np.mean(spectral_rolloff)

    # Темп
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo.item() if hasattr(tempo, 'item') else tempo

    return {
        "mfcc_mean": mfcc_mean,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_rolloff_mean": spectral_rolloff_mean,
        "tempo": tempo
    }


def features_to_array(features: dict) -> np.ndarray:
    """
    Преобразует словарь признаков в одномерный массив numpy (43 признака)
    и добавляет размерность для модели: (1, 43).
    """
    return np.hstack([
        features["mfcc_mean"],
        features["spectral_centroid_mean"],
        features["spectral_rolloff_mean"],
        features["tempo"]
    ]).reshape(1, -1)