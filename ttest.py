import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel

pca = PCA(n_components=4)

data_stage1 = pd.read_csv('data_stage1.csv')
data_stage1 = data_stage1.iloc[:, 1:]
data_stage1 = data_stage1.T
data_stage1 = (data_stage1 - data_stage1.mean()) / data_stage1.std()
pca_components_stage1 = pd.DataFrame(pca.fit_transform(data_stage1))
pca_components_stage1 = pca_components_stage1.T
pca_components_stage1.index = ['Фактор 1', 'Фактор 2', 'Фактор 3', 'Фактор 4']

data_stage2 = pd.read_csv('data_stage2.csv')
data_stage2 = data_stage2.iloc[:, 1:]
data_stage2 = data_stage2.T
data_stage2 = (data_stage2 - data_stage2.mean()) / data_stage2.std()
pca_components_stage2 = pd.DataFrame(pca.fit_transform(data_stage2))
pca_components_stage2 = pca_components_stage2.T
pca_components_stage2.index = ['Фактор 1', 'Фактор 2', 'Фактор 3', 'Фактор 4']

data_stage3 = pd.read_csv('data_stage3.csv')
data_stage3 = data_stage3.iloc[:, 1:]
data_stage3 = data_stage3.T
data_stage3 = (data_stage3 - data_stage3.mean()) / data_stage3.std()
pca_components_stage3 = pd.DataFrame(pca.fit_transform(data_stage3))
pca_components_stage3 = pca_components_stage3.T
pca_components_stage3.index = ['Фактор 1', 'Фактор 2', 'Фактор 3', 'Фактор 4']

pca_components_stage1.to_csv('pca_component_stage1.csv')
pca_components_stage2.to_csv('pca_component_stage2.csv')
pca_components_stage3.to_csv('pca_component_stage3.csv')

# Среднее значение главных компонент для каждого этапа
mean_stage1 = pca_components_stage1.mean(axis=1)
mean_stage2 = pca_components_stage2.mean(axis=1)
mean_stage3 = pca_components_stage3.mean(axis=1)

# Объединение в одну таблицу
means = pd.concat([mean_stage1, mean_stage2, mean_stage3], axis=1)
means.columns = ['I Этап', 'II Этап', 'III Этап']
means.to_csv('pca_component_means.csv')

# Записать данные о изменении факторов в файл
changes = pd.DataFrame({
    'Фактор 1': [pca_components_stage2.loc['Фактор 1'].mean() / pca_components_stage1.loc['Фактор 1'].mean() * 100,
                 pca_components_stage3.loc['Фактор 1'].mean() / pca_components_stage2.loc['Фактор 1'].mean() * 100],
    'Фактор 2': [pca_components_stage2.loc['Фактор 2'].mean() / pca_components_stage1.loc['Фактор 2'].mean() * 100,
                 pca_components_stage3.loc['Фактор 2'].mean() / pca_components_stage2.loc['Фактор 2'].mean() * 100],
    'Фактор 3': [pca_components_stage2.loc['Фактор 3'].mean() / pca_components_stage1.loc['Фактор 3'].mean() * 100,
                 pca_components_stage3.loc['Фактор 3'].mean() / pca_components_stage2.loc['Фактор 3'].mean() * 100],
    'Фактор 4': [pca_components_stage2.loc['Фактор 4'].mean() / pca_components_stage1.loc['Фактор 4'].mean() * 100,
                 pca_components_stage3.loc['Фактор 4'].mean() / pca_components_stage2.loc['Фактор 4'].mean() * 100]
})

changes.index = ['Изменение I -> II (%)', 'Изменение II -> III (%)']
changes.to_csv('pca_component_changes.csv')

# Отдельно записать на каком уровне достоверности изменения статистически значимы
ttest = pd.DataFrame({
    'Фактор 1': [ttest_rel(pca_components_stage1.loc['Фактор 1'], pca_components_stage2.loc['Фактор 1']).pvalue,
                 ttest_rel(pca_components_stage2.loc['Фактор 1'], pca_components_stage3.loc['Фактор 1']).pvalue],
    'Фактор 2': [ttest_rel(pca_components_stage1.loc['Фактор 2'], pca_components_stage2.loc['Фактор 2']).pvalue,
                 ttest_rel(pca_components_stage2.loc['Фактор 2'], pca_components_stage3.loc['Фактор 2']).pvalue],
    'Фактор 3': [ttest_rel(pca_components_stage1.loc['Фактор 3'], pca_components_stage2.loc['Фактор 3']).pvalue,
                 ttest_rel(pca_components_stage2.loc['Фактор 3'], pca_components_stage3.loc['Фактор 3']).pvalue],
    'Фактор 4': [ttest_rel(pca_components_stage1.loc['Фактор 4'], pca_components_stage2.loc['Фактор 4']).pvalue,
                 ttest_rel(pca_components_stage2.loc['Фактор 4'], pca_components_stage3.loc['Фактор 4']).pvalue]
})

ttest.index = ['I -> II', 'II -> III']
ttest.to_csv('pca_component_ttest.csv')























