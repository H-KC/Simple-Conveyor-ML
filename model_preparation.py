import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# загрузка данных из файла
data = np.loadtxt('train/processed_data.csv', delimiter=',')

# создание и обучение модели машинного обучения
model = LinearRegression()
model.fit(data[:, :-1], data[:, -1])

# сохранение модели в файл
joblib.dump(model, 'model.pkl')
