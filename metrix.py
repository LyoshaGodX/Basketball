import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

def plot_correlation_matrix(data):
    """
    Функция для визуализации матрицы корреляций между факторами.
    :param data: DataFrame с факторами
    :return: None, визуализирует матрицу корреляций
    """
    fig, ax = plt.subplots(figsize=(40, 15))
    cax = ax.matshow(data.T.corr())
    fig.colorbar(cax)
    plt.xticks(range(len(data.T.columns)), range(1, len(data.T.columns) + 1), fontsize=15)
    plt.yticks(range(len(data.T.columns)), data.T.columns, fontsize=15)

    plt.tight_layout()
    plt.show()


def plot_pairwise_correlation(data1: DataFrame, data2: DataFrame):
    """
    Функция для визуализации отношения факторов между двумя этапами.
    :param data1: DataFrame с факторами на первом этапе
    :param data2: DataFrame с факторами на втором этапе
    :return: None, визуализирует отношение факторов между этапами
    """
    pass



def plot_factor_contributions(pca_components, feature_names):
    """
    Функция для визуализации вклада каждого признака в факторы.
    :param pca_components: DataFrame с факторами
    :param feature_names: Список названий признаков
    :return: None, визуализирует вклад признаков в факторы
    """
    n_factors = pca_components.shape[1]
    fig, axs = plt.subplots(1, n_factors, figsize=(20, 5))

    for i, ax in enumerate(axs):
        factor_contributions = np.abs(pca_components.iloc[:, i])
        factor_contributions.plot(kind='bar', ax=ax)
        ax.set_title(f'Фактор {i + 1}', fontsize=16)
        ax.set_xlabel('Признак', fontsize=14)
        ax.set_ylabel('Вклад', fontsize=14)
        ax.tick_params(axis='x', rotation=90)
        ax.set_xticklabels(feature_names, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_stacked_factor_contributions(pca_components, feature_names):
    """
    Функция для визуализации вклада каждого признака в факторы с помощью гистограммы с накоплением.
    :param pca_components: DataFrame с факторами
    :param feature_names: Список названий признаков
    :return: None, визуализирует вклад признаков в факторы
    """
    n_features = pca_components.shape[0]
    n_factors = pca_components.shape[1]
    feature_contributions = np.abs(pca_components).values

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(n_features)
    for i in range(n_factors):
        ax.bar(range(n_features), feature_contributions[:, i], bottom=bottom, label=f'Фактор {i + 1}')
        bottom += feature_contributions[:, i]

    ax.set_xlabel('Признак', fontsize=14)
    ax.set_ylabel('Вклад', fontsize=14)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=12)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()

