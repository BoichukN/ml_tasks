import numpy
import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

df = pandas.read_csv('wine.data', names=['Class',
                                         'Alcohol',
                                         'Malic acid',
                                         'Ash',
                                         'Alcalinity of ash',
                                         'Magnesium',
                                         'Total phenols',
                                         'Flavanoids',
                                         'Nonflavanoid phenols',
                                         'Proanthocyanins',
                                         'Color intensity',
                                         'Hue',
                                         'OD280/OD315 of diluted wines',
                                         'Proline'])

# Извлеките из данных признаки и классы.
# Класс записан в первом столбце (три варианта),
# признаки — в столбцах со второго по последний.
X_train = df[df.columns[1:]]
y_test = df[df.columns[0]]

# Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold).
# Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True).
# Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром random_state=42.
# В качестве меры качества используйте долю верных ответов (accuracy).
kv = KFold(n_splits=5, random_state=42, shuffle=True)


def accuracy_test(k_v, X, y):
    means = list()
    means_range = range(1, 51)

    for r in means_range:
        cls = KNeighborsClassifier(n_neighbors=r)
        score = cross_val_score(cls, X, y, cv=k_v, scoring='accuracy')
        score_mean = numpy.mean(score)
        means.append(score_mean)

    return pandas.DataFrame(means, means_range).mean(axis=1).sort_values(ascending=False)


# Найдите точность классификации на кросс-валидации для метода k ближайших соседей
# (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50.
# При каком k получилось оптимальное качество?
# Чему оно равно (число в интервале от 0 до 1)?
# Данные результаты и будут ответами на вопросы 1 и 2.
accuracy_clf_kv = accuracy_test(k_v=kv, X=X_train, y=y_test)

f = open('number_of_k.txt', 'w')
f.write('{}'.format(accuracy_clf_kv.index[0]))
f.close()

f = open('accuracy_clf_kv.txt', 'w')
f.write('{:0.2f}'.format(accuracy_clf_kv.values[0]))
f.close()

# Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale.
# Снова найдите оптимальное k на кросс-валидации.
X_train_standartize = scale(X_train)

# Какое значение k получилось оптимальным после приведения признаков к одному масштабу?
# Приведите ответы на вопросы 3 и 4.
accuracy_standartize = accuracy_test(k_v=kv, X=X_train_standartize, y=y_test)

f = open('number_of_k_standarize.txt', 'w')
f.write('{}'.format(accuracy_standartize.index[0]))
f.close()

f = open('accuracy_standartize.txt', 'w')
f.write('{:0.2f}'.format(accuracy_standartize.values[0]))
f.close()
