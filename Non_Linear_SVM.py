import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

"""
we use the data about articles that we want to decide that a article comes to our journal is popular to publish it or not.By using data mining all data extract and put in csv file
that we used here a cvs file. Informtion about each column is in Notepad file named information. 
Goal:Implement a model that helps us based on the information extract from an article that send to our journal,we decide it to publish or not. The artcile is publish if it has high
populaity which target shows us. We should use classification method as our y is categorical. All variables (x) are number(int, float).we use Linear SVM model here
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

#split data to X and target
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#scale the X variables and put in dataframe
X=pd.DataFrame(scale(X))
#split data to train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape )

#implement Non_linear SVM model use Radial basis function(rbf). gamma is become smaller --> linear, gamma is become larger --> non-linear
svm_rbf=svm.SVC(C=100,kernel="rbf",gamma=0.00001)
svm_rbf.fit(X_train,y_train)
#get the accuracy of the model
print(svm_rbf.score(X_test, y_test))
#put label on test data
print(svm_rbf.predict(X_test))
#do confusion_matrix and we have two rows as we have two classes
print('Linear','\n',confusion_matrix(y_test,svm_rbf.predict(X_test)))


#implement Polynomial SVM model. gamma is become smaller --> linear, gamma is become larger --> non-linear
##if we put degree = 1 and gamma=1 --> we get linear SVM
svm_poly=svm.SVC(C=100,kernel="poly",degree=2,gamma=0.001)
svm_poly.fit(X_train,y_train)
#get the accuracy of the model
print(svm_poly.score(X_test, y_test))
#put label on test data
print(svm_poly.predict(X_test))
#do confusion_matrix and we have two rows as we have two classes
print('Polynomial','\n',confusion_matrix(y_test,svm_poly.predict(X_test)))