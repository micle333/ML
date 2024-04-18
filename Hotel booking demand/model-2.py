# иморитирование всех необходимых библиотек
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
# Считываем тренировочные данные
train = pd.read_csv("E:/ItschoolSamsung/task2/train_final.csv")
test = pd.read_csv("E:/ItschoolSamsung/task2/test_final.csv")

train = train.drop(columns=['reservation_status_date', 'country', 'assigned_room_type', 'reserved_room_type'])
test = test.drop(columns=['reservation_status_date', 'country', 'assigned_room_type', 'reserved_room_type'])


train.loc[train['market_segment'] == 'Aviation', 'market_segment'] = 'Online TA'
test.loc[test['market_segment'] == 'Aviation', 'market_segment'] = 'Online TA'

train.loc[train['deposit_type'] == 'Refundable', 'deposit_type'] = 'No Deposit'
test.loc[test['deposit_type'] == 'Refundable', 'deposit_type'] = 'No Deposit'
#train = train.loc[train['market_segment'] != 'Aviation']
#test = test.loc[test['market_segment'] != 'Aviation']

#train = train.loc[train['deposit_type'] != 'Refundable']
#test = test.loc[test['deposit_type'] != 'Refundable']


# Закодируем часть категориальных признаков с помощью LabelEncoder()
# а часть с помощью GetDummies()

from sklearn import preprocessing

categ = train.loc[:, train.dtypes == object].columns

print(categ)

# Попробуем другой вариант кодировщика
train = pd.get_dummies(train, columns=categ)
test = pd.get_dummies(test, columns=categ)
print(test)
# Выделим вектор признаков и вектор ответов
X = train.drop(columns=['is_canceled'])
y = train['is_canceled']


# Создадим модель логистической регрессии
model = LogisticRegression()

# обучение модели
model.fit(X, y)

# предсказание ответов для тестовой выборки
answers_pred = model.predict(X)

from sklearn.metrics import accuracy_score, precision_score, recall_score

# answers_pred - ответы которые вернула модель для X_test
# y_test - это правильные ответы для X_test
print(f'Accuracy: {accuracy_score(y, answers_pred)}')
print(f'Precision: {precision_score(y, answers_pred)}')
print(f'Recall: {recall_score(y, answers_pred)}')

#test = test.drop(['Aviation', 'Refundable'])

y_pred_test = model.predict(test)
y_pred_test = pd.DataFrame(y_pred_test, columns=['is_canceled'])
y_pred_test = y_pred_test.reset_index()

y_pred_test.to_csv("E:/ItschoolSamsung/task2/3.csv", index=False)