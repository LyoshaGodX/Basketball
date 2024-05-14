import numpy as np
import pandas as pd
from generate_synthetic_data import generate_data, concatenate_data, check_randomness
from PCA import run_pca, save_pca_model, load_and_apply_pca_model, standardize_data, calculate_mean_pca_scores
from metrix import plot_correlation_matrix, plot_pairwise_correlation, plot_factor_contributions, \
    plot_stacked_factor_contributions

# -----------------------Первый этап-----------------------------

# Средние значения признаков
means_first_step = np.array([
    3.92, 37.20, 1.21, 52.20, 31.10, 174.70, 32.30, 12.40, 11.25, 4.25,
    20.25, 6.85, 15.25, 2.42, 4.57, 40.80, 4.33, 1.46, 13.70, 33.10,
    15.70, 47.20, 32.20, 3120.00
])

# Погрешности признаков (из вашей таблицы)
errors_first_step = np.array([
    0.05, 0.65, 0.08, 0.85, 0.54, 7.29, 0.78, 0.64, 0.73, 0.12,
    0.75, 0.54, 0.62, 0.04, 0.03, 0.69, 0.56, 0.07, 0.91, 1.52,
    0.42, 2.62, 1.82, 152.34
])

# Генерация синтетических данных для первого этапа
generate_data(means_first_step, errors_first_step, 'data_stage1.csv')

# -----------------------Второй этап-----------------------------

means_second_step = np.array([
    3.78, 42.45, 1.16, 49.62, 36.39, 178.30, 35.70, 15.60, 15.40, 3.94,
    24.38, 7.20, 17.35, 3.56, 6.44, 47.60, 6.52, 1.32, 12.80, 35.30,
    15.60, 50.40, 41.40, 3360.00
])

errors_second_step = np.array([
    0.05, 0.74, 0.07, 0.78, 0.65, 8.16, 0.82, 0.72, 0.77, 0.06,
    0.80, 0.78, 0.78, 0.06, 0.05, 0.78, 0.64, 0.06, 0.79, 1.43,
    0.49, 2.18, 2.91, 116.52
])

generate_data(means_second_step, errors_second_step, 'data_stage2.csv')

# -----------------------Третий этап-----------------------------

means_third_step = np.array([
    3.51, 48.50, 1.05, 45.38, 42.10, 186.40, 41.20, 18.40, 18.40, 3.41,
    27.63, 8.10, 21.45, 4.02, 8.06, 54.80, 8.34, 1.15, 11.20, 37.40,
    15.10, 56.70, 47.10, 3610.00
])

errors_third_step = np.array([
    0.02, 0.88, 0.04, 0.72, 0.74, 9.18, 0.96, 0.82, 0.87, 0.05,
    0.92, 0.94, 0.87, 0.09, 0.08, 0.89, 0.73, 0.04, 0.63, 1.35,
    0.64, 2.31, 2.52, 82.24
])

generate_data(means_third_step, errors_third_step, 'data_stage3.csv')

# Объединение данных из всех этапов
concatenate_data(['data_stage1.csv', 'data_stage2.csv', 'data_stage3.csv'], 'combined_data.csv')

# Применение PCA к объединенным данным
pca = run_pca(['data_stage1.csv', 'data_stage2.csv', 'data_stage3.csv'], n_components=4)

# Сохранение модели PCA
save_pca_model(pca, 'pca_model.pkl')

# Загрузка модели PCA и применение её к данным из каждого этапа
load_and_apply_pca_model(['data_stage1.csv', 'data_stage2.csv', 'data_stage3.csv'],
                         'pca_model.pkl',
                         ['pca_stage1.csv', 'pca_stage2.csv', 'pca_stage3.csv'],
                         ['score_stage1.csv', 'score_stage2.csv', 'score_stage3.csv'])

# -----------------------Анализ данных-----------------------------

# Загрузка данных
data_stage1 = pd.read_csv('data_stage1.csv', index_col=0)
data_stage2 = pd.read_csv('data_stage2.csv', index_col=0)
data_stage3 = pd.read_csv('data_stage3.csv', index_col=0)

pca_stage1 = pd.read_csv('pca_stage1.csv', index_col=0)
pca_stage2 = pd.read_csv('pca_stage2.csv', index_col=0)
pca_stage3 = pd.read_csv('pca_stage3.csv', index_col=0)

# Теперь удаляем из pca_stage* последнюю строку "Суммарный вклад"
pca_stage1 = pca_stage1.drop(pca_stage1.tail(1).index)
pca_stage2 = pca_stage2.drop(pca_stage2.tail(1).index)
pca_stage3 = pca_stage3.drop(pca_stage3.tail(1).index)

# # TODO: Проверка случайности данных
# check_randomness(data_stage1, ...)
# check_randomness(data_stage2, ...)
# check_randomness(data_stage3, ...)

# Визуализация матрицы корреляций
# plot_correlation_matrix(data_stage1)
# plot_correlation_matrix(data_stage2)
# plot_correlation_matrix(data_stage3)

# # TODO: Парные корреляции между факторами
# plot_pairwise_correlation((pca_stage1), (pca_stage2))
# plot_pairwise_correlation((pca_stage2), (pca_stage3))
# plot_pairwise_correlation((pca_stage1), (pca_stage3))

# # Визуализация вклада признаков в факторы
# plot_factor_contributions(standardize_data(pca_stage1), data_stage1.T.columns)
# plot_factor_contributions(standardize_data(pca_stage2), data_stage2.T.columns)
# plot_factor_contributions(standardize_data(pca_stage3), data_stage3.T.columns)

# # Визуализация вклада признаков в факторы с помощью гистограммы с накоплением
# plot_stacked_factor_contributions(standardize_data(pca_stage1), data_stage1.T.columns)
# plot_stacked_factor_contributions(standardize_data(pca_stage2), data_stage2.T.columns)
# plot_stacked_factor_contributions(standardize_data(pca_stage3), data_stage3.T.columns)

calculate_mean_pca_scores()
