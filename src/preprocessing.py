import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.scaler = StandardScaler()
        self.encoders = {}

    def load_data(self):
        """Загрузка данных из CSV или Excel."""
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"Данные загружены. Размер: {self.df.shape}")
        except FileNotFoundError:
            print("Файл не найден.")
        return self.df

    def clean_data(self):
        """Очистка от дубликатов и пропусков."""
        if self.df is not None:
            # Удаление дубликатов
            initial_rows = self.df.shape[0]
            self.df.drop_duplicates(inplace=True)
            print(f"Удалено дубликатов: {initial_rows - self.df.shape[0]}")

            # Обработка пропусков (пример: заполнение средним для чисел)
            # Для реальной задачи лучше выбирать стратегию под каждый столбец
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            imputer = SimpleImputer(strategy='mean')
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
            
            print("Пропуски обработаны.")
        return self.df

    def encode_categorical(self, columns):
        """Кодирование текстовых категорий в числа."""
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le  # Сохраняем, чтобы потом можно было декодировать
        print(f"Закодированы колонки: {columns}")
        return self.df

    def normalize_features(self, target_column):
        """Масштабирование числовых признаков (кроме целевой переменной)."""
        features = self.df.drop(columns=[target_column])
        numeric_features = features.select_dtypes(include=[np.number]).columns
        
        self.df[numeric_features] = self.scaler.fit_transform(self.df[numeric_features])
        print("Числовые признаки нормализованы.")
        return self.df

    def split_data(self, target_column, test_size=0.2):
        """Разделение на обучающую и тестовую выборки."""
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

# Пример использования (можно запустить этот файл отдельно)
if __name__ == "__main__":
    # Замените 'data.csv' на ваш файл
    processor = DataPreprocessor('data.csv')
    
    processor.load_data()
    processor.clean_data()
    
    # Укажите свои категориальные колонки
    processor.encode_categorical(['Category', 'City']) 
    
    # Укажите целевую колонку (то, что предсказываем)
    X_train, X_test, y_train, y_test = processor.split_data(target_column='Price')