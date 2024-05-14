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

        pca_array = pca.transform(standardized_data.T)

        save_pca_components(pca, data, output_file, pca_array)
        save_pca_scores(pca, data, output_scores_file, pca_array)


def save_pca_components(pca, data, file_name, pca_components=None):
    df_pca = pd.DataFrame(pca.components_.T, columns=[f'Фактор {i + 1}' for i in range(pca_components.shape[1])],
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
    mean_scores.to_csv('score_all_stages.csv', header=['Этап I', 'Этап II', 'Этап II'])


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
