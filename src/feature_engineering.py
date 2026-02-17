"""
Модуль для создания признаков для классификации
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle


class FeatureEngineer:
    """Класс для создания признаков"""

    def __init__(self):
        self.preprocessor = None
        self.feature_names = []

    def create_features(self, df: pd.DataFrame, target_col: str = 'level') -> Tuple[pd.DataFrame, pd.Series]:
        """Создает признаки для классификации"""
        features_df = df.copy()

        # Числовые признаки
        numerical_features = ['age', 'experience_years', 'salary']

        # Категориальные признаки
        categorical_features = ['city', 'education_level']

        # Бинарные признаки
        features_df['is_moscow'] = features_df['city'].apply(
            lambda x: 1 if isinstance(x, str) and 'москва' in x.lower() else 0
        )
        features_df['is_spb'] = features_df['city'].apply(
            lambda x: 1 if isinstance(x, str) and ('санкт-петербург' in x.lower() or 'спб' in x.lower()) else 0
        )

        binary_features = ['is_moscow', 'is_spb', 'has_car']

        # Создаем дополнительные признаки
        features_df['salary_per_year'] = features_df['salary'] / (features_df['experience_years'] + 1)
        features_df['age_experience_ratio'] = features_df['experience_years'] / (features_df['age'] + 1)

        additional_numerical = ['salary_per_year', 'age_experience_ratio']

        # Все признаки
        all_numerical = numerical_features + binary_features + additional_numerical

        # Целевая переменная
        y = features_df[target_col]

        # Признаки
        X = features_df[all_numerical + categorical_features]

        # Сохраняем имена признаков
        self.feature_names = all_numerical + categorical_features

        return X, y

    def create_preprocessor(self):
        """Создает пайплайн предобработки"""
        numerical_features = [f for f in self.feature_names if f in
                              ['age', 'experience_years', 'salary', 'salary_per_year',
                               'age_experience_ratio', 'is_moscow', 'is_spb', 'has_car']]
        categorical_features = [f for f in self.feature_names if f in ['city', 'education_level']]

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return self.preprocessor

    def save_preprocessor(self, path: str):
        """Сохраняет предобработчик"""
        with open(path, 'wb') as f:
            pickle.dump(self.preprocessor, f)

    def load_preprocessor(self, path: str):
        """Загружает предобработчик"""
        with open(path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        return self.preprocessor