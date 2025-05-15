import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Генерация синтетических данных
X = np.linspace(-3, 3, 1000).reshape(-1, 1)
y = X[:, 0] + 0.3 * np.sin(2 * X[:, 0]) + np.random.normal(0, 0.2 * np.abs(X[:, 0]), 1000)

# Создание модели с Dropout и двумя выходами (mean, log_var)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dropout(0.2),  # Dropout для эпистемической неопределенности
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2)  # Выходы: mean и log_var
])


# Функция потерь: Negative Log Likelihood
def nll_loss(y_true, y_pred):
    mean, log_var = tf.split(y_pred, 2, axis=-1)
    return 0.5 * tf.reduce_mean(tf.exp(-log_var) * (y_true - mean) ** 2 + log_var)

    model.compile(optimizer='adam', loss=nll_loss)
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)

    # Функция для расчета неопределенностей


def calculate_uncertainties(model, X_test, n_samples=100):
    all_means, all_log_vars = [], []

    # Многократный прогон с включенным Dropout
    for _ in range(n_samples):
        preds = model(X_test, training=True)  # training=True активирует Dropout
        mean, log_var = preds[:, 0], preds[:, 1]
        all_means.append(mean)
        all_log_vars.append(log_var)

    # Преобразуем в numpy массивы
    all_means = np.array(all_means)
    all_log_vars = np.array(all_log_vars)

    # Энтропия для каждого сэмпла (для гауссова распределения)
    entropy_per_sample = 0.5 * np.log(2 * np.pi * np.exp(1) * np.exp(all_log_vars))

    # Средняя алеаторная неопределенность (AU)
    au = np.mean(entropy_per_sample, axis=0)

    # Общая неопределенность (Total)
    total_variance = np.var(all_means, axis=0) + np.mean(np.exp(all_log_vars), axis=0)
    total_entropy = 0.5 * np.log(2 * np.pi * np.exp(1) * total_variance)

    # Эпистемическая неопределенность (MU = Total - AU)
    mu = total_entropy - au

    return au, mu


# Тестовые данные
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
au, mu = calculate_uncertainties(model, X_test)

# Визуализация
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], y, s=1, label='Данные')
plt.plot(X_test[:, 0], model.predict(X_test, verbose=0)[:, 0], 'r-', label='Предсказание')
plt.fill_between(X_test[:, 0], au, alpha=0.3, label='Aleatoric (AU)')
plt.fill_between(X_test[:, 0], mu, alpha=0.3, label='Epistemic (MU)')
plt.legend()
plt.title("Bayesian Aleatoric Decomposition")
plt.show()