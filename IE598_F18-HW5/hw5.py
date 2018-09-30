import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
import seaborn as sns

#part 1
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.shape
df_wine.info()
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
cols = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
   
# scatterplot matrix
sns.pairplot(df_wine[cols], size=5)
plt.tight_layout()
plt.savefig('scatterplot matrix.png',dpi=300)
plt.show()

#correlation matrix
cm = np.corrcoef(df_wine[cols].values.T)
sns.set(font_scale=0.2)
hm = sns.heatmap(cm,
cbar=True,
annot=True,
square=True,
fmt='.2f',
annot_kws={'size': 6},
yticklabels=cols,
xticklabels=cols)
plt.savefig('correlation matrix.png',dpi=300)
plt.show()

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#part2
# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train_std, y_train)
print('(LR_part2_train)score: %.10f' % (lr.score(X_train_std, y_train)))
print('(LR_part2_test)score: %.10f' % (lr.score(X_test_std, y_test)))

#SVM
svm = SVC()
svm.fit(X_train_std, y_train)
print('(SVM_part2_train)score: %.10f' % (svm.score(X_train_std, y_train)))
print('(SVM_part2_test)score: %.10f' % (svm.score(X_test_std, y_test)))

#part3
#PCA
# LogisticRegression
pca= PCA(n_components=13)
lr1 = LogisticRegression()
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
lr1.fit(X_train_pca, y_train)
print('(LR_part3_train)score: %.10f' % (lr1.score(X_train_pca, y_train)))
print('(LR_part3_test)score: %.10f' % (lr1.score(X_test_pca, y_test)))

#SVM
pca= PCA(n_components=13)
svm1 = SVC()
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
svm1.fit(X_train_pca, y_train)
print('(SVM_part3_train)score: %.10f' % (svm1.score(X_train_pca, y_train)))
print('(SVM_part3_test)score: %.10f' % (svm1.score(X_test_pca, y_test)))

#part4
#LDA
# LogisticRegression
lda= LDA(n_components=13)
lr2 = LogisticRegression()
X_train_lda=lda.fit_transform(X_train_std, y_train)
X_test_lda=lda.transform(X_test_std)
lr2.fit(X_train_lda, y_train)
print('(LR_part4_train)score: %.10f' % (lr2.score(X_train_lda, y_train)))
print('(LR_part4_test)score: %.10f' % (lr2.score(X_test_lda, y_test)))

#SVM
lda= LDA(n_components=13)
svm2 = SVC()
X_train_lda=lda.fit_transform(X_train_std, y_train)
X_test_lda=lda.transform(X_test_std)
svm2.fit(X_train_lda, y_train)
print('(SVM_part4_train)score: %.10f' % (svm2.score(X_train_lda, y_train)))
print('(SVM_part4_test)score: %.10f' % (svm2.score(X_test_lda, y_test)))

#part5
#KPCA
# LogisticRegression
gamma_space = np.logspace(-3, 0, 4)
scikit_kpca =KernelPCA(n_components=2,kernel='rbf')
for gamma in gamma_space:
    scikit_kpca.gamma=gamma
    X_train_kpca= scikit_kpca.fit_transform(X_train_std)
    X_test_kpca= scikit_kpca.transform(X_test_std)
    lr3 = LogisticRegression()
    lr3.fit(X_train_kpca, y_train)
    print('((gamma='+str(gamma)+' )LR_part5_train)score: %.10f' % (lr3.score(X_train_kpca, y_train)))
    print('((gamma='+str(gamma)+' )LR_part5_test)score: %.10f' % (lr3.score(X_test_kpca, y_test)))


#SVM
gamma_space = np.logspace(-3, 0, 4)
scikit_kpca =KernelPCA(n_components=2,kernel='rbf',gamma=0.1)
for gamma in gamma_space:
    scikit_kpca.gamma=gamma
    X_train_kpca= scikit_kpca.fit_transform(X_train_std)
    X_test_kpca= scikit_kpca.transform(X_test_std)
    svm3 = SVC()
    svm3.fit(X_train_kpca, y_train)
    print('((gamma='+str(gamma)+' )SVM_part5_train)score: %.10f' % (svm3.score(X_train_kpca, y_train)))
    print('((gamma='+str(gamma)+' )SVM_part5_test)score: %.10f' % (svm3.score(X_test_kpca, y_test)))



print("My name is {Ruozhong Yang}")
print("My NetID is: {ry8}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")