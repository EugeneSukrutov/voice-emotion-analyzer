"""
Загрузка обученной модели эмоций и предсказание.
"""
import os
import joblib
import numpy as np
from src.features import features_to_array

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_model.pkl')

_emotion_model = None
_emotion_classes = None


def load_models():
    """Загружает модель эмоций, если она ещё не загружена."""
    global _emotion_model, _emotion_classes
    if _emotion_model is None:
        if not os.path.exists(EMOTION_MODEL_PATH):
            raise FileNotFoundError(
                f"Модель эмоций не найдена: {EMOTION_MODEL_PATH}\n"
                "Сначала обучите модель в Jupyter и сохраните её в папку models/."
            )
        _emotion_model = joblib.load(EMOTION_MODEL_PATH)
        if hasattr(_emotion_model, 'classes_'):
            _emotion_classes = _emotion_model.classes_
        else:
            _emotion_classes = ['neutral', 'sad', 'angry', 'joy']


def predict_emotion(features_dict: dict) -> tuple:
    """
    Возвращает предсказанную эмоцию (строку) и вероятность.
    """
    load_models()
    X = features_to_array(features_dict)
    proba = _emotion_model.predict_proba(X)[0]
    pred_class = _emotion_model.predict(X)[0]
    confidence = np.max(proba)
    return pred_class, confidence
