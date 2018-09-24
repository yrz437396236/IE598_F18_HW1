import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

df = pd.read_csv('concrete.csv',sep=',')
df.shape
df.info()
cols = ['cement','slag','ash','water','superplastic','coarseagg','fineagg','age','strength']
#box plot
for i in ['cement','slag','ash','water','superplastic','coarseagg','fineagg','age','strength']:
    plt.figure()
    sns.boxplot(x=i,data=df)
    plt.savefig('box_plot_'+i+'.png', dpi=300)
# scatterplot matrix
sns.pairplot(df[cols], size=5)
plt.tight_layout()
plt.savefig('scatterplot matrix.png',dpi=300)
plt.show()

#correlation matrix
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=0.5)
hm = sns.heatmap(cm,
cbar=True,
annot=True,
square=True,
fmt='.2f',
annot_kws={'size': 10},
yticklabels=cols,
xticklabels=cols)
plt.savefig('correlation matrix.png',dpi=300)
plt.show()

X = df[['cement','slag','ash','water','superplastic','coarseagg','fineagg','age']].values
y = df['strength'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)

#LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train,
c='steelblue', marker='o', edgecolor='white',
label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
c='limegreen', marker='s', edgecolor='white',
label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
plt.xlim([-10, 90])
plt.savefig('LinearRegression.png', dpi=300)
plt.show()
print('(LR)MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('(LR)R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
print('(LR)Slope: %.3f' % reg.coef_[0])
print('(LR)Intercept: %.3f' % reg.intercept_)

#RidgeRegression
alpha_space = np.logspace(-3, 0, 4)
ridge = Ridge(normalize=True)
# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha   
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    plt.scatter(y_train_pred, y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.savefig('Ridge(alpha='+str(alpha)+' ).png', dpi=300)
    plt.show()
    print('Ridgealpha: %.3f' %(alpha))
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    print('Slope: %.3f' % ridge.coef_[0])
    print('Intercept: %.3f' % ridge.intercept_)
    
#LessoRegression
alpha_space = np.logspace(-6, -3, 4)    
lasso = Lasso(normalize=True)
# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    lasso.alpha = alpha   
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    plt.scatter(y_train_pred, y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.savefig('Lasso(alpha='+str(alpha)+' ).png', dpi=300)
    plt.xlim([-10, 90])
    plt.show()
    print('Lassoalpha: %.6f' %(lasso.alpha))
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    print('Slope: %.3f' % lasso.coef_[0])
    print('Intercept: %.3f' % lasso.intercept_)

#Elastic Net regression
l1_ratio_space = np.logspace(-3, 0, 4)    
elanet = ElasticNet(alpha=1) 
# Compute scores over range of alphas
for l1_ratio in l1_ratio_space:

    # Specify the alpha value to use: ridge.alpha

    elanet.l1_ratio = l1_ratio   
    elanet.fit(X_train, y_train)
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)
    plt.scatter(y_train_pred, y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=90, color='black', lw=2)
    plt.xlim([-10, 90])
    plt.savefig('Elastic_Net(l1_ratio='+str(l1_ratio)+' ).png', dpi=300)
    plt.show()
    print('Elastic_Net_l1_ratio: %.6f' %(elanet.l1_ratio))
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    print('Slope: %.3f' % elanet.coef_[0])
    print('Intercept: %.3f' % elanet.intercept_)
    
print("My name is Ruozhong Yang")
print("My NetID is: ry8")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")





