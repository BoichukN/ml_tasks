import re

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# 1. Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
sex_counts = data['Sex'].value_counts()
f = open('female_male.txt', 'w')
f.write('{} {}'.format(sex_counts['male'], sex_counts['female']))
f.close()


# 2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).
surv_counts = data['Survived'].value_counts()
surv_percent = 100.0 * surv_counts[1] / surv_counts.sum()
f = open('survived_percent.txt', 'w')
f.write("{:0.2f}".format(surv_percent))
f.close()


# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).
pclass_counts = data['Pclass'].value_counts()
pclass_percent = 100.0 * pclass_counts[1] / pclass_counts.sum()
f = open('pclass_percent.txt', 'w')
f.write("{:0.2f}".format(pclass_percent))
f.close()


# 4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.
f = open('age_of_passengers.txt', 'w')
f.write("{:0.2f} {:0.2f}".format(data['Age'].mean(), data['Age'].median()))
f.close()


# 5. Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
corr = data['SibSp'].corr(data['Parch'])
f = open('correlation.txt', 'w')
f.write("{:0.2f}".format(corr))
f.close()


# 6. Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name)
# его личное имя (First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию. Попробуйте вручную разобрать
# несколько значений столбца Name и выработать правило для извлечения имен, а также разделения их на женские и мужские.
def clean_name(name):
    # Перше слово до коми це фамілія
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)

    # Якщо є лапки то імя в них
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)
    # Видаляємо статус
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)

    # Беремо перше слово що залишилось і видаляємо лапки
    name = name.split(' ')[0].replace('"', '')
    return name


names = data[data['Sex'] == 'female']['Name'].map(clean_name)
name_counts = names.value_counts()

f = open('most_popular_name.txt', 'w')
f.write(name_counts.head(1).index.values[0])
f.close()
