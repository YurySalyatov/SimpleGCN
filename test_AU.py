import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Создание синтетических данных (с добавлением размерности)
X = np.linspace(-3, 3, 1000).reshape(-1, 1)  # Форма (1000, 1)
y = X[:, 0] + 0.3 * np.sin(2 * X[:, 0]) + np.random.normal(0, 0.2 * np.abs(X[:, 0]), 1000)

# 2. Обучение модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),  # Явно задаём input_shape
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, batch_size=32, verbose=0)  # Теперь X имеет форму (1000, 1)

# 3. Функция TTIP (с исправлением размерности)
def calculate_aleatoric_uncertainty(model, X_test, n_samples=100, noise_std=0.2):
    predictions = []
    X_test = X_test.reshape(-1, 1)  # Добавляем размерность
    for _ in range(n_samples):
        X_perturbed = X_test + np.random.normal(0, noise_std, size=X_test.shape)
        pred = model.predict(X_perturbed, verbose=0).flatten()
        predictions.append(pred)
    return np.mean(predictions, axis=0), np.var(predictions, axis=0)

# 4. Тестовые данные (также с размерностью)
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)  # Форма (100, 1)
mean_pred, aleatoric_var = calculate_aleatoric_uncertainty(model, X_test)

# 5. Визуализация
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], y, s=1, label='Данные')
plt.plot(X_test[:, 0], mean_pred, 'r-', label='Предсказание')
plt.fill_between(
    X_test[:, 0],
    mean_pred - 2*np.sqrt(aleatoric_var),
    mean_pred + 2*np.sqrt(aleatoric_var),
    alpha=0.3,
    label='Неопределенность'
)
plt.legend()
plt.show()