import numpy as np
import pandas as pd

# Средние значения признаков (из вашей таблицы)
means = np.array([
    3.92, 37.20, 1.21, 52.20, 31.10, 174.70, 32.30, 12.40, 11.25, 4.25,
    20.25, 6.85, 15.25, 2.42, 4.57, 40.80, 4.33, 1.46, 13.70, 33.10,
    15.70, 47.20, 32.20, 3120.00
])

# Погрешности признаков (из вашей таблицы)
errors = np.array([
    0.05, 0.65, 0.08, 0.85, 0.54, 7.29, 0.78, 0.64, 0.73, 0.12,
    0.75, 0.54, 0.62, 0.04, 0.03, 0.69, 0.56, 0.07, 0.91, 1.52,
    0.42, 2.62, 1.82, 152.34
])

# Названия признаков
feature_names = [
    "Бег 20 м (с)",
    "Десять восьмерок (сек)",
    "Комбинированное упражнение (мин)",
    "Переменный скоростной бег 100 м (сек)",
    "Прыжок в высоту с места (см)",
    "Прыжок в длину с места (см)",
    "Брюшной пресс (раз)",
    "Отжимания от пола (раз за 30 сек)",
    "Подтягивание на низкой перекладине (раз)",
    "Бег 1 000 м (мин)",
    "Броски мяча в корзину с разных точек 40 бросков (кол. попаданий)",
    "Бросок набивного мяча (м)",
    "Ведение мяча с закрытыми глазами в кругу (к-во кас)",
    "Оценка ощущения времени (балл)",
    "Передачи мяча (кол-во)",
    "Штрафные броски %",
    "Ведения мяча с изменением направления и бросками (кол.попад)",
    "Передвижение в защитной стойке 100 м (мин)",
    "Челночный бег 3х10 м (сек)",
    "Динамометрия доминантной рукой (кг)",
    "Частота дыхания (кол-во раз)",
    "Проба Штанге (с)",
    "Проба Генчи (с)",
    "Жизненная емкость легких (мл)"
]

# Количество наблюдений (спортсменов)
n_samples = 20

# Создаем пустой DataFrame
df = pd.DataFrame(columns=feature_names)

# Генерируем наблюдения для каждого признака
for i in range(len(means)):
    observation_i = np.random.normal(means[i], errors[i], n_samples)
    df[feature_names[i]] = observation_i

df = df.transpose()

# Сохраняем DataFrame в CSV файл
df.to_csv("generated_data_first_step.csv", index=True)

print("Таблица успешно сохранена в файле generated_data.csv")