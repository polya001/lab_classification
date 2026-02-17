"""
Вспомогательные функции для проекта
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional
import matplotlib

matplotlib.use('TkAgg')  # Для отображения графиков в Windows


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution", save_path: Optional[str] = None) -> None:
    """Визуализирует распределение классов"""
    class_counts = pd.Series(y).value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.index, class_counts.values, color=['blue', 'green', 'red'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Уровень разработчика', fontsize=12)
    plt.ylabel('Количество резюме', fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    plt.grid(axis='y', alpha=0.3)

    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        percentage = (height / len(y)) * 100
        plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{int(height)} ({percentage:.1f}%)',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Выводим статистику
    print("\nРаспределение классов:")
    print("-" * 40)
    for level, count in class_counts.items():
        percentage = count / len(y) * 100
        print(f"  {level:10s}: {count:4d} резюме ({percentage:6.1f}%)")


def analyze_errors(X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray,
                   original_data: pd.DataFrame) -> pd.DataFrame:
    """Анализирует ошибки классификации"""
    errors = original_data.copy()
    errors['true_level'] = y_test
    errors['pred_level'] = y_pred
    errors['is_correct'] = errors['true_level'] == errors['pred_level']

    # Ошибки по классам
    error_by_class = errors[~errors['is_correct']].groupby('true_level').size()
    print("\nОшибки по истинным классам:")
    print(error_by_class)

    # Распространенные ошибки
    common_errors = pd.crosstab(errors['true_level'], errors['pred_level'])
    print("\nМатрица ошибок:")
    print(common_errors)

    return errors


def save_results(results: Dict[str, Any], path: str) -> None:
    """Сохраняет результаты в JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)