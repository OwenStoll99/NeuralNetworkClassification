import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#retrieve data for processing
dataset = pd.read_csv("wdbc.data",header=None) 

#separate data into dependent and independant variables, X for independent, y for dependent
X = pd.DataFrame(dataset.iloc[:, 2:33].values)
y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
#transform binary data into 0 or 1 (in this case transform M or B)
y = labelencoder.fit_transform(y)

#split into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#scale data to normalize within range
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#build the network now that data is ready
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#input layer
classifier.add(Dense(12, activation = 'relu', input_dim = 30))

#second hidden layer
classifier.add(Dense(6, activation = 'relu'))

#add output layer
classifier.add(Dense(1, activation = 'sigmoid'))

#compile ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#train the ANN
hist = classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#print error vs epoch data
plt.figure()
plt.plot(hist.history['loss'])
plt.title('Model Error vs. Epoch')
plt.ylabel('error')
plt.xlabel('epoch')
plt.show()

#print accuracy vs epoch data
plt.plot(hist.history['accuracy'])
plt.title('Model Accuracy vs. Epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

#predict results of test data
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print (accuracy_score(y_test,y_pred))