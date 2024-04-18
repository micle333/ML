# иморитирование всех необходимых библиотек
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Считываем тренировочные данные
train = pd.read_csv("E:/ItschoolSamsung/task1/train.csv")
formula_train = pd.read_csv("E:/ItschoolSamsung/task1/formula_train.csv")
test = pd.read_csv("E:/ItschoolSamsung/task1/test.csv")
formula_test = pd.read_csv("E:/ItschoolSamsung/task1/formula_test.csv")

formula_train = formula_train.drop(columns=['critical_temp'])
train_full = pd.concat([train, formula_train], axis=1)
# Удалим из данных ненужную колонку 'material'
train_full.drop(columns=['material'], inplace=True)

# Выделим из набора данных вектор признаков и вектор ответов
X = train_full.drop(columns=['critical_temp'])
y = train_full['critical_temp']
test_full = pd.concat([test, formula_test], axis=1)
# Удалим из данных ненужную колонку 'material'
test_full.drop(columns=['material'], inplace=True)





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# импортируем необходмимые библиотеки
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Создадим модель линейной регрессии
model = LinearRegression()
# Обучим линейную регрессию на тренировочном стандартизированном наборе данных
# На этапе обучения алгоритм рассчитывает веса признаков и строит линейное уравнение регрессии
model.fit(X, y)
# Создадим массив названия признаков
features = test_full.columns
coeff_df = pd.DataFrame(model.coef_, columns=['Coefficient'])
coeff_df['features'] = features

y_pred = model.predict(X)

print('Mean Absolute Error:', mean_absolute_error(y, y_pred))
print('Mean Squared Error:', mean_squared_error(y, y_pred))

y_pred_test = model.predict(test_full)
y_pred_test = pd.DataFrame(y_pred_test, columns=['critical_temp'])
y_pred_test = y_pred_test.reset_index()
y_pred_test.to_csv("predict.csv", index=False)

