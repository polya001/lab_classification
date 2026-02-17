"""
Модуль для предобработки данных hh.ru
"""
import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Класс для предобработки данных hh.ru"""

    def __init__(self):
        self.it_keywords = [
            # Разработчики
            'разработчик', 'developer', 'программист', 'software',
            'инженер-программист', 'web', 'frontend', 'backend', 'fullstack',
            'data scientist', 'аналитик данных', 'data engineer',
            'системный администратор', 'сетевой инженер', 'devops',
            'тестировщик', 'qa', 'тест', 'инженер по тестированию',
            # Технические специалисты
            'архитектор', 'team lead', 'тимлид', 'техлид',
            'ml engineer', 'машинное обучение', 'ai', 'ai engineer'
        ]

        self.junior_keywords = [
            'junior', 'младший', 'стажёр', 'intern', 'начинающий'
        ]

        self.middle_keywords = [
            'middle', 'средний', 'опытный', 'ведущий'
        ]

        self.senior_keywords = [
            'senior', 'старший', 'ведущий', 'главный', 'head', 'руководитель',
            'архитектор', 'team lead', 'тимлид'
        ]

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загружает данные из CSV файла"""
        return pd.read_csv(filepath, encoding='utf-8')

    def filter_it_resumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрует IT-резюме по ключевым словам"""
        def is_it_position(position):
            if isinstance(position, str):
                position_lower = position.lower()
                return any(keyword in position_lower for keyword in self.it_keywords)
            return False

        # Проверяем два столбца с должностями
        mask = (
            df['Ищет работу на должность:'].apply(is_it_position) |
            df['Последеняя/нынешняя должность'].apply(is_it_position)
        )

        it_df = df[mask].copy()
        print(f"Найдено IT-резюме: {len(it_df)} из {len(df)}")
        return it_df

    def extract_age(self, text: str) -> Optional[float]:
        """Извлекает возраст из строки 'Пол, возраст'"""
        if isinstance(text, str):
            # Ищем паттерн "XX лет/год/года"
            pattern = r'(\d+)\s*(лет|год|года)'
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return np.nan

    def extract_experience(self, text: str) -> Optional[float]:
        """Извлекает опыт работы в годах"""
        if isinstance(text, str):
            # Ищем паттерн "X лет Y месяцев" или "X лет"
            years_pattern = r'(\d+)\s*лет'
            months_pattern = r'(\d+)\s*месяц'

            years = re.search(years_pattern, text)
            months = re.search(months_pattern, text)

            total_years = 0
            if years:
                total_years += float(years.group(1))
            if months:
                total_years += float(months.group(1)) / 12

            return total_years
        return np.nan

    def extract_salary(self, salary_str: str) -> Optional[float]:
        """Извлекает зарплату в рублях"""
        if isinstance(salary_str, str):
            # Удаляем пробелы и 'руб.'
            numbers = re.findall(r'\d+', salary_str.replace(' ', ''))
            if numbers:
                return float(numbers[0])
        return np.nan

    def extract_city(self, city_str: str) -> str:
        """Извлекает основной город"""
        if isinstance(city_str, str):
            # Берем первый город до запятой
            city = city_str.split(',')[0].strip()
            return city
        return 'Не указано'

    def extract_education(self, edu_str: str) -> str:
        """Определяет уровень образования"""
        if isinstance(edu_str, str):
            if 'высшее' in edu_str.lower():
                return 'Высшее'
            elif 'среднее' in edu_str.lower() or 'спту' in edu_str.lower():
                return 'Среднее специальное'
            elif 'неполное' in edu_str.lower():
                return 'Неполное высшее'
            else:
                return 'Другое'
        return 'Не указано'

    def determine_level(self,
                        position: str,
                        experience: float,
                        current_position: str = None) -> str:
        """
        Определяет уровень разработчика по должности и опыту
        """
        position_text = str(position).lower() + ' ' + str(current_position or '').lower()

        # Проверяем ключевые слова в названии должности
        for keyword in self.senior_keywords:
            if keyword in position_text:
                return 'senior'

        for keyword in self.middle_keywords:
            if keyword in position_text:
                return 'middle'

        for keyword in self.junior_keywords:
            if keyword in position_text:
                return 'junior'

        # Если нет ключевых слов, определяем по опыту
        if pd.isna(experience):
            return 'unknown'

        if experience < 2:
            return 'junior'
        elif experience < 5:
            return 'middle'
        else:
            return 'senior'

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Основная функция предобработки"""
        processed_df = df.copy()

        # Извлекаем признаки
        processed_df['age'] = processed_df['Пол, возраст'].apply(self.extract_age)
        processed_df['experience_years'] = processed_df['Опыт (двойное нажатие для полной версии)'].apply(self.extract_experience)
        processed_df['salary'] = processed_df['ЗП'].apply(self.extract_salary)
        processed_df['city'] = processed_df['Город'].apply(self.extract_city)
        processed_df['education_level'] = processed_df['Образование и ВУЗ'].apply(self.extract_education)
        processed_df['has_car'] = processed_df['Авто'].apply(
            lambda x: 1 if isinstance(x, str) and 'автомобиль' in x.lower() else 0
        )

        # Определяем уровень
        processed_df['level'] = processed_df.apply(
            lambda row: self.determine_level(
                row['Ищет работу на должность:'],
                row['experience_years'],
                row['Последеняя/нынешняя должность']
            ),
            axis=1
        )

        # Очищаем от пропущенных значений
        processed_df = processed_df.dropna(subset=['age', 'experience_years', 'salary'])

        # Удаляем unknown уровни
        processed_df = processed_df[processed_df['level'] != 'unknown']

        print(f"После очистки осталось резюме: {len(processed_df)}")
        return processed_df