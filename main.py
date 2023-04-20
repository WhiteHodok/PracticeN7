import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Загрузка данных
df = pd.read_csv('data.csv')
# Удаление столбца "id", так как он не нужен для классификации
df.drop('id', axis=1, inplace=True)
# Разделение на признаки и метки классов
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
# Преобразование меток классов в числа (0 - доброкачественная, 1 - злокачественная)
y = np.where(y == 'M', 1, 0)
# Разделение на обучающую, валидационную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
# Определение модели
model = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_dim=30)
])
# Компиляция модели
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Обучение модели
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=16)
# Оценка качества модели на тестовой выборке
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Точность на тестовой выборке:', test_acc)
