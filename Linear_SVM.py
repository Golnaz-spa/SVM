import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score

"""
we use the data about articles that we want to decide that a article comes to our journal is popular to publish it or not. By using data mining all data extract and put in a csv file
that we used here a cvs file. Informtion about each column is in Notepad file named information. 
Goal:Implement a model that helps us based on the information extract from an article that send to our journal, we decide it to publish or not. The artcile is publish if it has high
populaity which target shows us. We should use classification method as our y is categorical. All variables (x) are number(int, float). we use Linear SVM model here
"""
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)

#read csv data
df=pd.read_csv("E:/file/OnlineNewsPopularity-c.csv")
print(df.shape)

print(df.head(2))
print(df.info())
#find unique values of target(H:high popularity, L:low popularity)
print(np.unique(df.target))
#find propertion of each label in this dataset
print((df['target'].value_counts())/df.shape[0])

#we want to know that how is the scale of dataset so use std for it
print(df.agg(np.std, axis=0))

#split data to X and target
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#scale the X variables and put in dataframe
X=pd.DataFrame(scale(X))
#split data to train and test data. 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape )

#implement SVM model.kernel="linear" means that we want to use Linear SVM
svm_linear=svm.SVC(C=50,kernel="linear")
svm_linear.fit(X_train,y_train)
#put label on test data
print(svm_linear.predict(X_test))
#get the accuracy of the model
print(svm_linear.score(X_test,y_test))
#do confusion_matrix and we have two rows as we have two classes
print('Linear','\n',confusion_matrix(y_test,svm_linear.predict(X_test)))



#we can improve the model by using Cross validation for find best value for C. Aftr that use the best C in the model and get scor, and confusion_matrix
nneighbors=np.arange(10,60,10)
#cvscores is numpy.ndarray for inserting the C and score of it
cvscores=np.empty((len(nneighbors),2))
counter=-1
# Perform 10-fold cross validation in order to find the best C
for k in nneighbors:
    counter = counter + 1
    svm_linear = svm.SVC(C=k, kernel="linear")
    cvscores[counter, :] = np.array([k, np.mean(cross_val_score(svm_linear, X_train, y_train, cv=10))])
    print(cvscores[counter, :])
print("maximum of C and maximum score is:", cvscores[np.argmax(cvscores[:, 1]), :])
best_C= int(cvscores[np.argmax(cvscores[:, 1]), :][0])

svm_linearcross=svm.SVC(C=best_C,kernel="linear")
svm_linearcross.fit(X_train,y_train)
#put label on test data
print(svm_linearcross.predict(X_test))
#get the accuracy of the model
print(svm_linearcross.score(X_test,y_test))
#do confusion_matrix and we have two rows as we have two classes
print('Linear','\n',confusion_matrix(y_test,svm_linearcross.predict(X_test)))



# we put the name of the model that we want to use in future. here is svm_rbf
import pickle

file_name = "E:/file/svm_news.pkl"
#Save the model - we dump it here meant change the file to binary
with open(file_name, 'wb') as file:
    pickle.dump(svm_linear, file)


# Read the model and use it in future -by using this pickle.load read the model and put it in pickle_model
with open(file_name, 'rb') as file:
    pickle_model = pickle.load(file)

pickle_model.predict(X_test)