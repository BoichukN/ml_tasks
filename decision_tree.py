import pandas
from sklearn.tree import DecisionTreeClassifier

# читаємо файл і індексуємо по колонці 'PassengerId'
df = pandas.read_csv('titanic.csv', index_col='PassengerId')

# залишаємо лише колонки 'Pclass', 'Sex', 'Age' & 'Fare'
dc = df.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'])

# замінюємо рядкові значення у рядку стать на 0 та 1 для жінок та чоловіків відповідно
dc.loc[dc['Sex'] == 'female', 'Sex'] = 0
dc.loc[dc['Sex'] == 'male', 'Sex'] = 1

# видаляємо дані що пропущені
dc.dropna(inplace=True)

# створюємо тренувальну вибірку і цілюву зміну
X = dc[['Pclass', 'Sex', 'Age', 'Fare']]
y = dc['Survived']

# навчання дерева з параметром random_state=241
clf = DecisionTreeClassifier(random_state=241, criterion='gini')
clf = clf.fit(X, y)

# визначення важливості признаків
importance = clf.feature_importances_
out = pandas.DataFrame({'col_name': importance}, index=X.columns).sort_values(by='col_name', ascending=False)
print(out)
