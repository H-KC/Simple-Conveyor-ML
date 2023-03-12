import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

# загрузка данных из файла
data = np.loadtxt('test/test_data.csv', delimiter=',')

# загрузка модели из файла
model = joblib.load('model.pkl')

# выполнение предобработки данных
scaler = joblib.load('train/scaler.pkl')
processed_data = scaler.transform(data)

# получение предсказаний модели
y_pred = model.predict(processed_data[:, :-1])

# вычисление метрик качества модели
mse = mean_squared_error(processed_data[:, -1], y_pred)
print('Mean squared error:', mse)
