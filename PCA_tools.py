import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle


def load_data(file_name):
    return pd.read_csv(file_name, index_col=0)


def load_and_apply_pca_model(data_files, model_file, output_files, output_scores_files):
    pca = load_pca_model(model_file)

    for data_file, output_file, output_scores_file in zip(data_files, output_files, output_scores_files):
        data = load_data(data_file)
        standardized_data = standardize_data(data)

        pca_array = pca.fit_transform(standardized_data.T)

        save_pca_components(pca, data, output_file, pca_array)
        save_pca_scores(pca, data, output_scores_file, pca_array)


def pca_2_components_model_plot(data_files, pca):
    """
    Функция принимает на вход список файлов с исходными данными. Для каждого файла строит модель PCA с двумя компонентами. Далее функция строит график, на котором точками отмечена закономерность первой компоненты от второй. Все данные представлены на одном графике, каждый файл отмечен уникальным цветом.
    """
    plt.figure(figsize=(10, 7))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, data_file in enumerate(data_files):
        data = load_data(data_file)
        standardized_data = standardize_data(data)
        pca_array = pca.transform(standardized_data.T)

        plt.scatter(pca_array[:, 0], pca_array[:, 1], c=colors[i % len(colors)], label=data_file)

    plt.title('Факторы', fontsize=16)
    plt.xlabel('Главная компонента 1', fontsize=14)
    plt.ylabel('Главная компонента 2', fontsize=14)
    plt.legend()
    plt.show()


def save_pca_components(pca, data, file_name, pca_components=None):
    df_pca = pd.DataFrame(normalize_weights(pca.components_.T),
                          columns=[f'Фактор {i + 1}' for i in range(pca_components.shape[1])],
                          index=data.index)
    df_pca = df_pca.rename_axis('Признаки', axis=0)
    df_pca.loc['Суммарный вклад'] = df_pca.abs().sum()
    df_pca.to_csv(file_name, index=True)


def save_pca_scores(pca, data, file_name, pca_scores=None):
    df_pca_scores = pd.DataFrame(pca_scores.T, columns=[f'Наблюдение {i + 1}' for i in range(pca_scores.shape[0])],
                                 index=[f'Главная компонента {i + 1}' for i in range(pca_scores.shape[1])])
    df_pca_scores.to_csv(file_name)


def standardize_data(data):
    return (data - data.mean()) / data.std()


def normalize_weights(weights):
    normalized_weights = np.zeros_like(weights)
    for i in range(weights.shape[1]):
        col = weights[:, i]
        min_val = np.min(col)
        max_val = np.max(col)
        range_val = max_val - min_val

        if range_val == 0:
            normalized_weights[:, i] = np.full_like(col, 0.5)
        else:
            normalized_weights[:, i] = (col - min_val) / range_val

    return normalized_weights


def save_pca_model(pca, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(pca, f)


def load_pca_model(file_name) -> PCA:
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def run_pca(data_files, n_components):
    combined_data = pd.concat([load_data(file) for file in data_files], axis=1)
    standardized_data = standardize_data(combined_data)
    pca = PCA(n_components=n_components).fit(standardized_data.T)
    return pca


def calculate_mean_pca_scores():
    # Чтение данных из файлов
    scores_stage1 = pd.read_csv('score_stage1.csv', index_col=0)
    scores_stage2 = pd.read_csv('score_stage2.csv', index_col=0)
    scores_stage3 = pd.read_csv('score_stage3.csv', index_col=0)

    # Вычисление среднего значения для каждой компоненты PCA по всем файлам
    mean_scores = pd.concat([scores_stage1.mean(axis=1),
                             scores_stage2.mean(axis=1),
                             scores_stage3.mean(axis=1)], axis=1)

    # Сохранение результатов в файл
    mean_scores.to_csv('score_all_stages.csv', header=['Этап I', 'Этап II', 'Этап III'])


def plot_correlation_matrix(data):
    fig, ax = plt.subplots(figsize=(40, 15))
    cax = ax.matshow(data.T.corr())
    fig.colorbar(cax)
    plt.xticks(range(len(data.T.columns)), range(1, len(data.T.columns) + 1), fontsize=20)
    plt.yticks(range(len(data.T.columns)), data.T.columns, fontsize=30)
    plt.show()


def plot_explained_variance_ratio(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100)
    plt.xticks(range(len(pca.explained_variance_ratio_)), range(1, len(pca.explained_variance_ratio_) + 1))
    plt.xlabel('Количество факторов')
    plt.ylabel('Процент сохраненной информации')
    plt.show()


def plot_component_variance(pca):
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100)
    plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    plt.xlabel('Фактор')
    plt.ylabel('Вклад фактора в общую дисперсию, %')
    plt.show()

def plot_elbow_method(data_files):
    combined_data = pd.concat([load_data(file) for file in data_files], axis=1)
    standardized_data = standardize_data(combined_data)
    explained_variances = []
    for n in range(1, min(standardized_data.shape) + 1):
        pca = PCA(n_components=n)
        pca.fit(standardized_data.T)
        explained_variances.append(sum(pca.explained_variance_ratio_))

    plt.plot(range(1, min(standardized_data.shape) + 1), explained_variances, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()