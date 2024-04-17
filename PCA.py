import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

# Загрузка данных
df = pd.read_csv("data_first_step.csv", index_col=0)

# Корреляционная матрица
fig, ax = plt.subplots(figsize=(40, 15))  # Увеличиваем размер холста
cax = ax.matshow(df.T.corr())
fig.colorbar(cax)
plt.xticks(range(len(df.T.columns)), range(1, 25), fontsize=20)
plt.yticks(range(len(df.T.columns)), df.T.columns, fontsize=30)
plt.show()

# Стандартизация данных
df = (df - df.mean()) / df.std()

# Коммулянта объяснимой дисперсии
plt.plot(np.cumsum(PCA(n_components=8).fit(df.T).explained_variance_ratio_) * 100)
plt.xticks(range(0, 8), range(1, 9))
plt.xlabel('Количество факторов')
plt.ylabel('Процент сохраненной информации')
plt.show()

# Применение PCA
pca = PCA(n_components=4).fit(df.T)
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Столбчатая диаграмма дисперсии главных компонент
plt.bar(range(1, 5), pca.explained_variance_ratio_ * 100)
plt.xticks(range(1, 5))
plt.xlabel('Фактор')
plt.ylabel('Вклад фактора в общую дисперсию, %')
plt.show()

# Запрись в таблицу
df_pca = pd.DataFrame(pca.components_.T, columns=['Фактор 1', 'Фактор 2', 'Фактор 3', 'Фактор 4'], index=df.index)
df_pca = df_pca.rename_axis('Признаки', axis=0)

# Добавление строки с суммарным вкладом каждого признака
df_pca.loc['Суммарный вклад'] = df_pca.abs().sum()

df_pca.to_csv('PCA_first_step.csv', index=True)