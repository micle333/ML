import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("E:/ItschoolSamsung/task3/Physical_Activity_Monitoring_unlabeled.csv")
df.drop(columns=['handTemperature', 'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 'handAcc16_3', 'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
                'handGyro1', 'handGyro2', 'handGyro3', 'handMagne1', 'handMagne2', 'handMagne3', 'handOrientation1', 'handOrientation2',
                'handOrientation3', 'handOrientation4'], inplace=True)


col_missing = ['chestTemperature', 'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 'chestAcc6_1', 'chestAcc6_2',
               'chestAcc6_3', 'chestGyro1', 'chestGyro2', 'chestGyro3', 'chestMagne1',
               'chestMagne2', 'chestMagne3', 'chestOrientation1', 'chestOrientation2',
               'chestOrientation3', 'chestOrientation4', 'ankleTemperature',
               'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 'ankleAcc6_1',
               'ankleAcc6_2', 'ankleAcc6_3', 'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
               'ankleMagne1', 'ankleMagne2', 'ankleMagne3', 'ankleOrientation1',
               'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

for i in col_missing:
    df[i].fillna(df[i].mean(), inplace=True)
df = df.drop(columns=['timestamp'])

#print(df.isna().sum())


from sklearn.cluster import KMeans

n_clusters = 4

clusterer = KMeans(n_clusters=n_clusters)

cluster_labels = clusterer.fit_predict(df)

# inertia
#print(clusterer.inertia_)

# inertia_df = pd.DataFrame(data=[], index=range(3, 12), columns=['inertia'])
# #
# for n_clusters in range(3, 12):
#      clusterer = KMeans(n_clusters=n_clusters, random_state=42)
#      cluster_labels = clusterer.fit_predict(df)

     # inertia
     #inertia_df.loc[n_clusters] = clusterer.inertia_

# print(inertia_df.plot())

predictions = pd.DataFrame(cluster_labels, columns=['activityID'])
predictions = predictions.reset_index()

indexa = predictions['activityID'].unique()

for index, row in predictions.iterrows():
     if row['activityID'] == int(indexa[0]):
         predictions.loc[index, 'activityID'] = 1
     if row['activityID'] == int(indexa[1]):
         predictions.loc[index, 'activityID'] = 2
     if row['activityID'] == int(indexa[2]):
         predictions.loc[index, 'activityID'] = 3
     if row['activityID'] == int(indexa[3]):
         predictions.loc[index, 'activityID'] = 4
#     if row['activityID'] == int(indexa[4]):
#         predictions.loc[index, 'activityID'] = 5
#     if row['activityID'] == int(indexa[5]):
#         predictions.loc[index, 'activityID'] = 6
#     if row['activityID'] == int(indexa[6]):
#         predictions.loc[index, 'activityID'] = 7
#     if row['activityID'] == int(indexa[7]):
#         predictions.loc[index, 'activityID'] = 8
#     if row['activityID'] == int(indexa[8]):
#         predictions.loc[index, 'activityID'] = 9
#     if row['activityID'] == int(indexa[9]):
#         predictions.loc[index, 'activityID'] = 10
#     if row['activityID'] == int(indexa[10]):
#         predictions.loc[index, 'activityID'] = 11

#predictions = predictions.drop(predictions.columns[[0]], axis = 1)

print(predictions)

predictions.to_csv("E:/ItschoolSamsung/task3/solution.csv", index=False)




# from sklearn.metrics import silhouette_score
#
# silhouette_avg = silhouette_score(df, cluster_labels)
#
#
# from yellowbrick.cluster import silhouette_visualizer
#
# X = df.sample(frac=0.1)
# print(X.shape)
# silhouette_visualizer(KMeans(3, random_state=42), X, colors='yellowbrick')