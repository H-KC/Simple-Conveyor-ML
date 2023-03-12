import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Создаем папки, если их нет
if not os.path.exists('train'):
    os.makedirs('train')

if not os.path.exists('test'):
    os.makedirs('test')

# Генерируем случайные данные
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Добавляем аномалии и шумы
X[0] = 1000
X[1, 2] = np.nan
X[2, 4] = 100
y[3] = 100

# Разбиваем данные на обучающую и тестовую выборки
train_size = 80
test_size = 20

X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Сохраняем данные в папках train и test
pd.DataFrame(X_train).to_csv('train/X_train.csv', index=False)
pd.DataFrame(y_train).to_csv('train/y_train.csv', index=False)

pd.DataFrame(X_test).to_csv('test/X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('test/y_test.csv', index=False)
