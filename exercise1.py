#python3 -m venv .venv
#install scikit-learn
#pip3 install openpyxl
#"Gerald Clark CMSI 630 exercise1"

import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

def my_nearest_neighbor():
    heart_data = 'my_data1/heart-attack-tests.csv'

    #df_h = pd.read_csv(heart_data, delimiter=';', low_memory=False, na_values='?').drop(['unnamed 0'],axis=1)  
    df_h = pd.read_csv(heart_data, usecols=["Age","Sex", "Chest Pain","Rest BP","Cholesterol","Blood Sugar","Rest ECG","Max Heart Rate",
                                            "Exercise Angina","Old Peak","Slope","Discolored Vessels","Thalessemia"], encoding='utf-8', na_values=['?',''], index_col=False)
    df_h1 = df_h.fillna(-1)
    df_h2 = df_h1[df_h1["Discolored Vessels"]>=0]
    print(df_h2)
    
    df_h1.to_excel("heart_data4.xlsx")
    #heart data used for testing, very few values
    #X_test = df_h2.iloc[:,:-1].values
    #y_test = df_h2.iloc[:,13].values

    #Cleveland Data
    cleveland_data = 'my_data1/cleveland.csv'
    df_c = pd.read_csv(cleveland_data, usecols=["Age","Sex", "Chest Pain","Rest BP","Cholesterol","Blood Sugar","Rest ECG","Max Heart Rate",
                                            "Exercise Angina","Old Peak","Slope","Discolored Vessels","Thalessemia", "Disease Value"], na_values=['?',''], index_col=False)
    df_c1  = df_c.fillna(-1)
    df_c2 = df_c1.drop_duplicates(keep="first")
    #df_c2["Age"] = df_c2["Age"].astype(float)
    df_c3 = df_c2.sort_values(by=['Disease Value'], ascending=False)
    df_c3a = df_c3[df_c3["Disease Value"]<=4]
    df_c4a = df_c3a[df_c3a["Age"] !=0]
    df_c4 = df_c4a[df_c4a["Thalessemia"]>=0]
    df_c5a = df_c4[df_c4["Blood Sugar"]<=1]
    df_c5b = df_c5a[df_c5a["Max Heart Rate"]>=0]
    df_c5 = df_c5b[df_c5b["Sex"]>=0]
    df_c5.to_excel("cleveland_data4.xlsx")

    ff = df_c5.shape
    print(ff)
    #print(df_c5.shape)

    print(df_c5["Disease Value"].value_counts())
    #sns.set_palette("pastel")
    #sns.color_palette("rocket")
    #sns.countplot(x="Disease Value", data=df_c5, hue="Disease Value", palette="PuBu_d")
    #plt.show()

    plt.scatter(x=df_c5["Age"][df_c5["Disease Value"]==0], y=df_c5["Max Heart Rate"][(df_c5["Disease Value"]==0)], c="green")
    plt.scatter(x=df_c5["Age"][df_c5["Disease Value"]==1], y=df_c5["Max Heart Rate"][(df_c5["Disease Value"]==1)], c="black")
    plt.scatter(x=df_c5["Age"][df_c5["Disease Value"]==2], y=df_c5["Max Heart Rate"][(df_c5["Disease Value"]==2)], c="blue")
    plt.scatter(x=df_c5["Age"][df_c5["Disease Value"]==3], y=df_c5["Max Heart Rate"][(df_c5["Disease Value"]==3)], c="grey")
    plt.scatter(x=df_c5["Age"][df_c5["Disease Value"]==4], y=df_c5["Max Heart Rate"][(df_c5["Disease Value"]==4)], c="brown")
    plt.legend(["no risk","slight risk", "moderate risk", "elevated risk", "extreme risk"])
    plt.xlabel("Age")
    plt.ylabel("Maximum Heart Rate")
    plt.show()

    #clevelend data used for training, ~300 values
    X = df_c5.iloc[:,:-1].values
    y = df_c5.iloc[:,13].values

    #split data into a training and test set. Using test set against training set to find acccuracy of model using algorithm k nearest neighbor.
    X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state=0)

    #scale data down on x axis
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2, weights='uniform') # euclidean, minkowski, manhattan
    classifier = classifier.fit(X_train,y_train)

    #make predictions then checks for accuracy
    y_prediction = classifier.predict(X_test)
    print(classification_report(y_test, y_prediction))
    print(confusion_matrix(y_test, y_prediction))

    cl_2 = KNeighborsClassifier(n_neighbors = 3, metric ='minkowski', p = 2, weights='uniform')
    cl_2 = cl_2.fit(X_train,y_train)
    #prediction function from sklearn
    y_pred_2 = cl_2.predict(X_test)
    #check accuracy
    print(classification_report(y_test, y_pred_2))
    print(confusion_matrix(y_test, y_pred_2))

    cl_3 = KNeighborsClassifier(n_neighbors = 5, metric ='minkowski', p = 2, weights='uniform')
    cl_3 = cl_3.fit(X_train,y_train)
    y_pred_3 = cl_3.predict(X_test)
    print(classification_report(y_test, y_pred_3))
    print(confusion_matrix(y_test, y_pred_3))

    cl_4 = KNeighborsClassifier(n_neighbors = 9, metric ='minkowski', p = 2, weights='uniform')
    cl_4 = cl_4.fit(X_train,y_train)
    y_pred_4 = cl_4.predict(X_test)
    print(classification_report(y_test, y_pred_4))
    print(confusion_matrix(y_test, y_pred_4))

    cl_5 = KNeighborsClassifier(n_neighbors = 15, metric ='minkowski', p = 2, weights='uniform')
    cl_5 = cl_5.fit(X_train,y_train)
    y_pred_5 = cl_5.predict(X_test)
    print(classification_report(y_test, y_pred_5))
    print(confusion_matrix(y_test, y_pred_5))

    classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'euclidean', p = 2, weights='uniform') # euclidean, minkowski, manhattan
    classifier = classifier.fit(X_train,y_train)

    #make predictions then checks for accuracy
    y_prediction = classifier.predict(X_test)
    print(classification_report(y_test, y_prediction))
    print(confusion_matrix(y_test, y_prediction))

    cl_2 = KNeighborsClassifier(n_neighbors = 3, metric ='euclidean', p = 2, weights='uniform')
    cl_2 = cl_2.fit(X_train,y_train)
    #prediction function from sklearn
    y_pred_2 = cl_2.predict(X_test)
    #check accuracy
    print(classification_report(y_test, y_pred_2))
    print(confusion_matrix(y_test, y_pred_2))

    cl_3 = KNeighborsClassifier(n_neighbors = 5, metric ='euclidean', p = 2, weights='uniform')
    cl_3 = cl_3.fit(X_train,y_train)
    y_pred_3 = cl_3.predict(X_test)
    print(classification_report(y_test, y_pred_3))
    print(confusion_matrix(y_test, y_pred_3))

    cl_4 = KNeighborsClassifier(n_neighbors = 9, metric ='euclidean', p = 2, weights='uniform')
    cl_4 = cl_4.fit(X_train,y_train)
    y_pred_4 = cl_4.predict(X_test)
    print(classification_report(y_test, y_pred_4))
    print(confusion_matrix(y_test, y_pred_4))

    cl_5 = KNeighborsClassifier(n_neighbors = 15, metric ='euclidean', p = 2, weights='uniform')
    cl_5 = cl_5.fit(X_train,y_train)
    y_pred_5 = cl_5.predict(X_test)
    print(classification_report(y_test, y_pred_5))
    print(confusion_matrix(y_test, y_pred_5))

    classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'manhattan', p = 2, weights='uniform') # euclidean, minkowski, manhattan
    classifier = classifier.fit(X_train,y_train)

    #make predictions then checks for accuracy
    y_prediction = classifier.predict(X_test)
    print(classification_report(y_test, y_prediction))
    print(confusion_matrix(y_test, y_prediction))

    cl_2 = KNeighborsClassifier(n_neighbors = 3, metric ='manhattan', p = 2, weights='uniform')
    cl_2 = cl_2.fit(X_train,y_train)
    #prediction function from sklearn
    y_pred_2 = cl_2.predict(X_test)
    #check accuracy
    print(classification_report(y_test, y_pred_2))
    print(confusion_matrix(y_test, y_pred_2))

    cl_3 = KNeighborsClassifier(n_neighbors = 5, metric ='manhattan', p = 2, weights='uniform')
    cl_3 = cl_3.fit(X_train,y_train)
    y_pred_3 = cl_3.predict(X_test)
    print(classification_report(y_test, y_pred_3))
    print(confusion_matrix(y_test, y_pred_3))

    cl_4 = KNeighborsClassifier(n_neighbors = 9, metric ='manhattan', p = 2, weights='uniform')
    cl_4 = cl_4.fit(X_train,y_train)
    y_pred_4 = cl_4.predict(X_test)
    print(classification_report(y_test, y_pred_4))
    print(confusion_matrix(y_test, y_pred_4))

    cl_5 = KNeighborsClassifier(n_neighbors = 15, metric ='manhattan', p = 2, weights='uniform')
    cl_5 = cl_5.fit(X_train,y_train)
    y_pred_5 = cl_5.predict(X_test)
    print(classification_report(y_test, y_pred_5))
    print(confusion_matrix(y_test, y_pred_5))

    #Manhattan distance with 9 and 15 nearest Neighbors is optimal.


    print("___completed___")

my_nearest_neighbor()











#  https://towardsdatascience.com/heart-disease-uci-diagnosis-prediction-b1943ee835a7
# https://www.kdnuggets.com/2019/07/classifying-heart-disease-using-k-nearest-neighbors.html/2








