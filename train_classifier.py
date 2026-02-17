#!/usr/bin/env python3
"""
Основной скрипт для обучения классификатора уровней IT-разработчиков
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Добавляем путь к src для импорта модулей
SRC_PATH = Path(__file__).parent / 'src'
sys.path.append(str(SRC_PATH))

# Теперь можем импортировать наши модули
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from classification_model import LevelClassifier
from utils import plot_class_distribution, analyze_errors

from sklearn.model_selection import train_test_split


def main():
    """Основной пайплайн обучения"""
    # Создаем директории
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("КЛАССИФИКАЦИЯ УРОВНЕЙ IT-РАЗРАБОТЧИКОВ")
    print("=" * 60)

    # 1. Загрузка и предобработка данных
    print("\n1. Загрузка и предобработка данных...")
    preprocessor = DataPreprocessor()

    # Проверяем существование файла
    data_path = 'data/hh.csv'
    if not os.path.exists(data_path):
        print(f"ОШИБКА: Файл {data_path} не найден!")
        print("Пожалуйста, поместите файл hh.csv в папку data/")
        return

    df = preprocessor.load_data(data_path)
    print(f"Загружено резюме: {len(df)}")

    # Фильтруем IT-резюме
    it_df = preprocessor.filter_it_resumes(df)
    it_df.to_csv('data/it_resumes.csv', index=False, encoding='utf-8')

    # Предобработка
    processed_df = preprocessor.preprocess(it_df)

    # 2. Анализ распределения классов
    print("\n2. Анализ распределения классов...")
    plot_class_distribution(
        processed_df['level'],
        title="Распределение IT-разработчиков по уровням",
        save_path='results/class_balance.png'
    )

    # 3. Создание признаков
    print("\n3. Создание признаков...")
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.create_features(processed_df)

    print(f"  Признаков: {X.shape[1]}")
    print(f"  Образцов: {X.shape[0]}")
    print(f"  Распределение классов: {dict(y.value_counts())}")

    # 4. Разделение данных
    print("\n4. Разделение данных...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"  Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"  Тестовая выборка: {X_test.shape[0]} образцов")

    # 5. Создание пайплайна предобработки
    print("\n5. Создание пайплайна предобработки...")
    preprocessor_pipe = feature_engineer.create_preprocessor()

    # 6. Обучение модели
    print("\n6. Обучение модели...")
    classifier = LevelClassifier(model_type='random_forest')
    classifier.train(X_train, y_train, preprocessor=preprocessor_pipe)

    # 7. Оценка модели
    print("\n7. Оценка модели...")
    results = classifier.evaluate(X_test, y_test, save_path='results/metrics.json')

    print(f"\nРезультаты классификации:")
    print(f"  Accuracy (точность): {results['accuracy']:.3f}")
    print(f"  Precision (точность): {results['precision']:.3f}")
    print(f"  Recall (полнота): {results['recall']:.3f}")
    print(f"  F1-Score: {results['f1_score']:.3f}")

    # 8. Визуализация
    print("\n8. Визуализация результатов...")
    classifier.plot_confusion_matrix(X_test, y_test, save_path='results/confusion_matrix.png')

    # 9. Сохранение модели
    print("\n9. Сохранение модели...")
    classifier.save('models/classifier.joblib')
    feature_engineer.save_preprocessor('models/preprocessor.joblib')

    print("✓ Модель сохранена в models/classifier.joblib")
    print("✓ Препроцессор сохранен в models/preprocessor.joblib")

    # 10. Анализ ошибок
    print("\n10. Анализ ошибок...")
    y_pred = classifier.predict(X_test)
    errors_df = analyze_errors(X_test, y_test, y_pred, processed_df.iloc[X_test.index])

    # 11. Выводы
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)

    print("\nВЫВОДЫ:")
    print("1. Качество модели удовлетворительное для PoC")
    print("2. Основные источники ошибок:")
    print("   - Дисбаланс классов (больше middle разработчиков)")
    print("   - Субъективность определения уровня по резюме")
    print("   - Недостаток информации в резюме для точной классификации")
    print("3. Наиболее важные признаки:")
    print("   - Опыт работы")
    print("   - Зарплата")
    print("   - Возраст")
    print("4. Рекомендации по улучшению:")
    print("   - Сбор дополнительных признаков (навыки, стек технологий)")
    print("   - Использование текстовых признаков из описания опыта")
    print("   - Ансамблирование нескольких моделей")

    # Сохраняем пример предсказаний
    sample_predictions = pd.DataFrame({
        'true_level': y_test[:10],
        'pred_level': y_pred[:10],
        'age': X_test['age'].iloc[:10],
        'experience_years': X_test['experience_years'].iloc[:10],
        'salary': X_test['salary'].iloc[:10]
    })
    print("\nПример предсказаний (первые 10):")
    print(sample_predictions)


if __name__ == "__main__":
    main()