import numpy as np
from sklearn.preprocessing import StandardScaler

# загрузка данных из файла
data = np.loadtxt('train/train_data.csv', delimiter=',')

# выполнение предобработки данных
scaler = StandardScaler()
processed_data = scaler.fit_transform(data)

# сохранение предобработанных данных в новый файл
np.savetxt('train/processed_data.csv', processed_data, delimiter=',')
