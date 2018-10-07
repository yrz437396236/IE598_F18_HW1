from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
#part1

iris = datasets.load_iris()
X, y = iris.data, iris.target
list1=[]
list2=[]
for i in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    print('test_score(random_state='+str(i)+')= %.3f' %(tree.score(X_test, y_test)))
    print('train_score(random_state='+str(i)+')= %.3f' %(tree.score(X_train, y_train)))
    list1.append(tree.score(X_test, y_test))
    list2.append(tree.score(X_train, y_train))
print('test_mean=%.5f'%np.mean(list1))
print('test_std=%.5f'%np.std(list1))
print('train_mean=%.5f'%np.mean(list2))
print('train_std=%.5f'%np.std(list2))

#part2
tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,stratify=y)
tree.fit(X_train, y_train)
cv_scores = cross_val_score(tree,X_train,y_train,cv=10)
print(cv_scores)
print('train_cv(random_state='+str(i)+')mean=%.5f'%np.mean(cv_scores))
print('train_cv(random_state='+str(i)+')std=%.5f'%np.std(cv_scores))
print(tree.score(X_test,y_test))
print("My name is Ruozhong Yang ")
print("My NetID is: ry8")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
