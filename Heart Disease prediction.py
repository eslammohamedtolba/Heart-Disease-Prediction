# import required modules
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np



# loading dataset 
Heart_disease_dataset = pd.read_csv("heart_disease_data.csv")
# show the dataset
Heart_disease_dataset.head()
Heart_disease_dataset.tail()
# show the dataset shape
Heart_disease_dataset.shape
# show some statistical info about the dataset
Heart_disease_dataset.describe()

# show some relations between the output and features in the dataset
Heart_disease_dataset.groupby('target').mean()
# check if there is any non(missing) values in the dataset
Heart_disease_dataset.isnull().sum()
# show the all info about the dataset
Heart_disease_dataset.info()  
# show the groups of target and will count the rapetition of each group and plot this repetitions
Heart_disease_dataset['target'].value_counts()
sns.catplot(x = 'target',data=Heart_disease_dataset,kind='count')

# find the correlation between featuers in the dataset
correlation_values = Heart_disease_dataset.corr()
# plot the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation_values,square=True,cbar=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap="Blues")

# plot the relation between target && cp features
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
sns.barplot(x='cp',y='target',data = Heart_disease_dataset)
plt.subplot(1,2,2)
sns.countplot(x='cp',hue='target',data = Heart_disease_dataset)
plt.show()




# split the data into input and label data
X = Heart_disease_dataset.drop(columns=['target'],axis=1)
Y = Heart_disease_dataset['target']
print(X)
print(Y)
# split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,stratify=Y,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)




# create the model and train it
LRModel = LogisticRegression()
LRModel.fit(x_train,y_train)
# make the model make prediction on train data
predicted_train_values = LRModel.predict(x_train)
# make the model make prediction on test data
predicted_test_values = LRModel.predict(x_test)
# avaluate model train prediction
Avaluation_prediction_train = accuracy_score(predicted_train_values,y_train)
# avaluate model test prediction
Avaluation_prediction_test = accuracy_score(predicted_test_values,y_test)
# show the accuracy values
print(Avaluation_prediction_train,Avaluation_prediction_test)




# Making a predictive system
input_data=(68,1,2,180,274,1,0,150,1,1.6,1,0,3)
# covert input data into 1D numpy array
input_array = np.array(input_data)
# convert 1D input array into 2D
input_2D_array = input_array.reshape(1,-1)
# make the model predict the output
if LRModel.predict(input_2D_array)[0]==1:
    print("he/she has a heart disease")
else:
    print("he/she doesn't have a heart disease")

