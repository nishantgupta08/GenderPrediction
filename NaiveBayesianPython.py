import pandas as pd
data=pd.read_csv("data.csv")

#####Splitting Dataset Into Train and Test##############
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(data['Gender'], 3, test_size=0.5, random_state=0)
for train_index, test_index in sss:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = data[train_index], data[test_index]

######Converting Categorical To Numerical##############
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#var_mod = ['LastLetter','LastTwoLetter','FirstLetter']
#for i in var_mod:
#    train[i] = le.fit_transform(train[i])
#    test[i] = le.fit_transform(test[i])
#
##Following Gaussian Distribution
#train.loc['LastLetter']['Gender'==1].plot.density()
#train['LastTwoLetter'].plot.density()  
#train['FirstLetter'].plot.density()
#
#
#import numpy as np
#from sklearn.naive_bayes import GaussianNB
#model = GaussianNB()
#
######Training Part##############
#numpyMatrix_train = train.as_matrix()
#x=np.array(numpyMatrix_train[:,[2,3,4]]).astype('int')
#y=np.array(numpyMatrix_train[:,1]).astype('int')
#model.fit(x,y)
#
######Testing Part##############
#numpyMatrix_test = test.as_matrix()
#X_test=np.array(numpyMatrix_test[:,[2,3,4]]).astype('int')
#Y_actual=np.array(numpyMatrix_test[:,1]).astype('int')
#Y_test= model.predict(X_test)
#
######Confusion Matrix##############
#from sklearn.metrics import confusion_matrix
#cnf_matrix = confusion_matrix(Y_actual, Y_test)
#print(cnf_matrix)
#
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(x, y)
#Y_test_decisiontree=clf.predict(X_test)
#cnf_matrix_decisontree = confusion_matrix(Y_actual, Y_test_decisiontree)
#print(cnf_matrix_decisontree)
