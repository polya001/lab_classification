"""
Модуль для классификации уровней разработчиков
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                           accuracy_score, precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


class LevelClassifier:
    """Классификатор уровней разработчиков"""

    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.class_weights = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              preprocessor: Optional[Any] = None, use_class_weights: bool = True) -> 'LevelClassifier':
        """Обучает модель классификации"""
        if preprocessor:
            self.preprocessor = preprocessor
            X_train_processed = preprocessor.fit_transform(X_train)
        else:
            X_train_processed = X_train

        # Вычисляем веса классов для борьбы с дисбалансом
        if use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            self.class_weights = dict(zip(classes, weights))
        else:
            self.class_weights = None

        # Выбираем модель
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=self.class_weights,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Обучаем модель
        self.model.fit(X_train_processed, y_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Предсказывает уровни"""
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X

        return self.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Возвращает вероятности классов"""
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = X

        return self.model.predict_proba(X_processed)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Оценивает модель и создает отчет"""
        y_pred = self.predict(X_test)

        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        # Отчет о классификации
        report = classification_report(y_test, y_pred, output_dict=True)

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

        # Сохраняем результаты
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # Сохраняем текстовый отчет
            text_report = classification_report(y_test, y_pred)
            with open(save_path.replace('.json', '_text.txt'), 'w', encoding='utf-8') as f:
                f.write(text_report)

        return results

    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: Optional[str] = None) -> None:
        """Визуализирует матрицу ошибок"""
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.model.classes_,
                   yticklabels=self.model.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save(self, model_path: str) -> None:
        """Сохраняет модель"""
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'preprocessor': self.preprocessor,
                'model_type': self.model_type,
                'class_weights': self.class_weights
            }, f)

    def load(self, model_path: str) -> 'LevelClassifier':
        """Загружает модель"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.preprocessor = data['preprocessor']
            self.model_type = data['model_type']
            self.class_weights = data['class_weights']
        return self