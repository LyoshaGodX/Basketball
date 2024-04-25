import numpy as np
import pandas as pd
import scipy.stats as stats



def generate_data(means: np.array, errors: np.array, file_name: str) -> None:
    """
    Функция генерации синтетических данных для анализа методом главных компонент.
    :param means: Средние значения признаков
    :param errors: Стандартные отклонения признаков
    :param file_name: Название файла для сохранения данных в формате CSV
    :return: none
    """

    # Названия признаков
    # feature_names = [
    #     "Бег 20 м (с)",
    #     "Десять восьмерок (сек)",
    #     "Комбинированное упражнение (мин)",
    #     "Переменный скоростной бег 100 м (сек)",
    #     "Прыжок в высоту с места (см)",
    #     "Прыжок в длину с места (см)",
    #     "Брюшной пресс (раз)",
    #     "Отжимания от пола (раз за 30 сек)",
    #     "Подтягивание на низкой перекладине (раз)",
    #     "Бег 1 000 м (мин)",
    #     "Броски мяча в корзину с разных точек 40 бросков (кол. попаданий)",
    #     "Бросок набивного мяча (м)",
    #     "Ведение мяча с закрытыми глазами в кругу (к-во кас)",
    #     "Оценка ощущения времени (балл)",
    #     "Передачи мяча (кол-во)",
    #     "Штрафные броски %",
    #     "Ведения мяча с изменением направления и бросками (кол.попад)",
    #     "Передвижение в защитной стойке 100 м (мин)",
    #     "Челночный бег 3х10 м (сек)",
    #     "Динамометрия доминантной рукой (кг)",
    #     "Частота дыхания (кол-во раз)",
    #     "Проба Штанге (с)",
    #     "Проба Генчи (с)",
    #     "Жизненная емкость легких (мл)"
    # ]
    feature_names = [
        "Бег20", "Восмерки", "Комбинированное", "СкоростнойБег100",
        "ПрыжокВысота", "ПрыжокДлина", "БрюшнойПресс", "Отжимания",
        "Подтягивание", "Бег1000", "БроскиМяча", "БросокМяча",
        "ВедениеМяча", "ОценкаВремени", "ПередачиМяча", "ШтрафныеБроски",
        "ВедениеМячаНаправление", "ПередвижениеЗащитнаяСтойка",
        "ЧелночныйБег", "Динамометрия", "ЧастотаДыхания", "ПробаШтанге",
        "ПробаГенчи", "ЕмкостьЛегких"
    ]

    # Количество наблюдений (спортсменов)
    n_samples = 20

    # Создаем пустой DataFrame
    df = pd.DataFrame(columns=feature_names)

    # Генерируем наблюдения для каждого признака, основываясь на среднем значении и погрешности
    for i in range(len(means)):
        observation_i = np.random.normal(means[i], errors[i], n_samples)

        # Округляем значения в зависимости от признака
        if feature_names[i] in ["Брюшной пресс (раз)", "Отжимания от пола (раз за 30 сек)",
                                "Подтягивание на низкой перекладине (раз)",
                                "Броски мяча в корзину с разных точек 40 бросков (кол. попаданий)",
                                "Ведение мяча с закрытыми глазами в кругу (к-во кас)", "Передачи мяча (кол-во)",
                                "Ведения мяча с изменением направления и бросками (кол.попад)",
                                "Частота дыхания (кол-во раз)", "Жизненная емкость легких (мл)"]:
            observation_i = np.round(observation_i, 0)  # Округляем до целого числа
        else:
            observation_i = np.round(observation_i, 2)  # Округляем до двух знаков после запятой

        # Устанавливаем ограничения для признака "Оценка ощущения времени (балл)"
        if feature_names[i] == "Оценка ощущения времени (балл)":
            # Округляем до целого числа в диапазоне от 1 до 5
            observation_i = np.round(np.clip(observation_i, 1, 5), 0)

        df[feature_names[i]] = observation_i

    # Транспонирование DataFrame для отображения признаков по строкам
    df = df.transpose()

    # Сохраняем DataFrame в CSV файл
    df.to_csv(file_name, index=True)


def check_randomness(synthetic_data, real_data):
    pass


def concatenate_data(file_names, output_file):
    """
    Функция для объединения данных из нескольких CSV файлов.
    :param file_names: Список имен файлов для объединения
    :param output_file: Имя выходного файла
    :return: None, сохраняет объединенные данные в указанный файл
    """
    data_frames = []
    for file_name in file_names:
        data_frames.append(pd.read_csv(file_name, index_col=0))

    combined_data = pd.concat(data_frames, axis=1, ignore_index=True)
    combined_data.to_csv(output_file, index=False)
