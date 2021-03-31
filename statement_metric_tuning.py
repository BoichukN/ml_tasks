import numpy
from pandas import DataFrame
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект, у которого признаки записаны в поле data,
# а целевой вектор — в поле target.
boston = load_boston()
y_train = boston.target

# Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
X_train = scale(boston.data)

# Качество оценивайте, как и в предыдущем задании,
# с помощью кросс-валидации по 5 блокам с random_state = 42,
# не забудьте включить перемешивание выборки (shuffle=True).
kv = KFold(n_splits=5, random_state=42, shuffle=True)


def quality(X, y, k_v):
    means = list()
    means_range = numpy.linspace(1, 10, 200)

    for p in means_range:
        r_k = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
        score = cross_val_score(r_k, X, y, cv=k_v, scoring='neg_mean_squared_error')
        score_mean = numpy.mean(score)
        means.append(score_mean)

    return DataFrame(means, means_range).mean(axis=1).sort_values(ascending=False)


# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом,
# чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace).
# Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' —
# данный параметр добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей.
# В качестве метрики качества используйте среднеквадратичную ошибку
# (параметр scoring='mean_squared_error' у cross_val_score;
# при использовании библиотеки scikit-learn версии 0.18.1 и выше необходимо указывать
# scoring='neg_mean_squared_error').
p_quality = quality(X_train, y_train, kv)

# Определите, при каком p качество на кросс-валидации оказалось оптимальным.
# Обратите внимание, что cross_val_score возвращает массив показателей качества по блокам;
# необходимо максимизировать среднее этих показателей.
# Это значение параметра и будет ответом на задачу.
with open('best_p_for_quality.txt', 'w') as f:
    f.write('{:0.1f}'.format(p_quality.index[0]))
