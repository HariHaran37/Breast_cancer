# %%
"""
# Problem Statement
"""

# %%
"""
The dataset has been taken from UCI machine learning repository. The main objective of the analysis is to perform classification of tumors i.e., benign(B) or malignant(M). A benign tumor is a tumor that does not invade its surrounding tissue or spread around the body. A malignant tumor is a tumor that may invade its surrounding tissue or spread around the body. This dataset consists of 569 rows and 32 columns.
"""

# %%
"""
Data Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
"""

# %%
"""
#### Attribute Informations
"""

# %%
"""
1) ID number

2) Diagnosis (M = malignant, B = benign)

3 to 32)                                           
3-12(mean)                                      
13-22(standard error)                                              
23-32(Worst)                                                  
 
Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)

b) texture (standard deviation of gray-scale values)

c) perimeter

d) area

e) smoothness (local variation in radius lengths)

f) compactness (perimeter^2 / area - 1.0)

g) concavity (severity of concave portions of the contour)

h) concave points (number of concave portions of the contour)

i) symmetry

j) fractal dimension ("coastline approximation" - 1)

"""

# %%
"""
# Breast Cancer prediction
"""

# %%
"""
#### Benign - B  Malignant - M

"""

# %%
import pandas as pd
#import pandas_profiling as pp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv("data/wdbc.data")# importing data
col=['ID','diagnosis','radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean','fractal_dimension_mean',
              'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
              'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst','fractal_dimension_worst']
df.columns =col # assign columns to data


# %%
"""
 we have 32 columns 
"""

# %%
"""
1 - ID
2 - classification data
3-32 - Numerical datas
"""

# %%
df.head() #checking if columns are added

# %%
df.info() #getting info from data like datatype , null values

# %%
df.describe() #getting described data

# %%
df.isnull().sum() #checking for null values

# %%
df['diagnosis'].value_counts() #counting B and M values to avoid baised model

# %%
"""
## Hist Plots
"""

# %%
for i in col[2:]:
    df[i].hist(bins=100)
    plt.show()
    plt.savefig('image/visual/Hist Plot/'+i+'.png')

# %%
"""
From the plot we can say columns are normally distributed
"""

# %%
"""
## Pairplots
"""

# %%
"""
Pairplot is used to check distribution of data
"""

# %%
means=sns.pairplot(df,hue='diagnosis',vars=col[2:12],diag_kind='hist',kind='scatter')
#fig=means.get_figure()
plt.savefig('image/visual/Pair Plot/plot.png')

# %%
"""
From the plot we can say perimeter,area and radius are correlated
"""

# %%
#se=sns.pairplot(df,hue='diagnosis',vars=col[12:22],diag_kind='kde',kind='scatter')
#plt.savefig('se.png')

# %%
#w=sns.pairplot(df,hue='diagnosis',vars=col[22:32],diag_kind='hist',kind='scatter')
#plt.savefig('w.png')

# %%
"""
## ECDF plots
"""

# %%
"""
ECDF plot is used to get cumulative distribution function of the distribution
"""

# %%
for i in col[2:32:10]:#getting radius based columns only(radius_mean,radius_se,radius_worst)
    sns.ecdfplot(df,x=i)
plt.legend(labels=col[2:32:10])
plt.xlabel('Radius')
plt.ylabel('Percentage')
plt.grid(color='grey')
plt.show()
plt.savefig('image/visual/ECDF plot/Radius.png')

# %%
for i in col[3:32:10]:
    sns.ecdfplot(df,x=i)
plt.legend(labels=col[3:32:10])
plt.grid(color='grey')
plt.xlabel('Texture')
plt.ylabel('Percentage')
plt.show()
plt.savefig('image/visual/ECDF plot/Texture.png')

# %%
for i in col[6:32:10]:
    sns.ecdfplot(df,x=i)
plt.legend(labels=col[6:32:10])
plt.xlabel('Smoothness')
plt.ylabel('Percentage')
plt.grid(color='grey')
plt.show()
plt.savefig('image/visual/ECDF plot/Smoothness.png')

# %%
for i in col[7:32:10]:
    sns.ecdfplot(df,x=i)
plt.legend(labels=col[7:32:10])
plt.xlabel('Campactness')
plt.ylabel('Percentage')
plt.grid(color='grey')
plt.show()
plt.savefig('image/visual/ECDF plot/Compactness.png')

# %%
for i in col[8:32:10]:
    sns.ecdfplot(df,x=i)
plt.legend(labels=col[8:32:10])
plt.xlabel('Concavity')
plt.ylabel('Percentage')
plt.grid(color='grey')
plt.show()
plt.savefig('image/visual/ECDF plot/Concavity.png')

# %%
for i in col[9:32:10]:
    sns.ecdfplot(df,x=i)
plt.legend(labels=col[9:32:10])
plt.xlabel('Concave_points')
plt.ylabel('Percentage')
plt.grid(color='grey')
plt.show()
plt.savefig('image/visual/ECDF plot/Concave_points.png')

# %%
for i in col[10:32:10]:
    sns.ecdfplot(df,x=i)
plt.legend(labels=col[10:32:10])
plt.xlabel('Symmetry')
plt.ylabel('Percentage')
plt.grid(color='grey')
plt.show()
plt.savefig('image/visual/ECDF plot/Symmetry.png')

# %%
for i in col[11:32:10]:
    sns.ecdfplot(df,x=i)
plt.legend(labels=col[11:32:10])
plt.xlabel('Fractional_dimension')
plt.ylabel('Percentage')
plt.grid(color='grey')
plt.show()
plt.savefig('image/visual/ECDF plot/Fractional_dimension.png')

# %%
"""
from plot we can know cumulative values of distribution in columns
"""

# %%
"""
# Bar plots
"""

# %%
"""
Bar plot is used to compare diagnosis with each other columns
"""

# %%
for i in range(2,12):
    fig = plt.figure(figsize = (12,8))
    sns.barplot(y = 'diagnosis', x = col[i], data = df)

    plt.title('Diagnosis vs '+col[i])
    plt.savefig('image/visual/Barplot/'+col[i]+'.jpeg')

# %%
"""
## Data Correlation
"""

# %%
mean=col[1:12] # getting mean values to future evaluvation
#because standard error and worst are metrics for mean values so mean is enough for our evalution

# %%
dataset = df[mean]# assigning data

# %%
corr=dataset.corr()#correlation values for dataset
mask = np.tril(np.ones_like(corr)) #getting only lower triangle

corr * mask

# %%
"""
### Heat map for correlation of dataset
"""

# %%
plt.figure(figsize=(20,12))
sns.heatmap(corr,mask=mask,annot=True,annot_kws={'size': 18})
plt.savefig('image/selection/heatmap_correlation.png')

# %%
"""
#### Function to get correlated columns
"""

# %%
def correlation(dataset, threshold):
    col_corr = set()  #The names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold: # for getting absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of columns
                col_corr.add(colname)
    return col_corr


# %%
cor_fe = correlation(dataset,0.8)#passing the dataset and thresold

# %%
cor_fe,len(cor_fe)

# %%
"""
Data column dropping
"""

# %%
dataset = dataset.drop(cor_fe,axis=1)#dropping correlated columns
print(dataset.columns)
y=dataset['diagnosis'] #assign diagnosis column to y
y.shape

# %%
dfm=dataset[dataset['diagnosis']=='M']#getting M data
dfb=dataset[dataset['diagnosis']=='B']#getting B data

# %%
"""
# Box plot
"""

# %%
"""
Box plot to see sections of data for each column in dataset and to identity outlier
"""

# %%
for i in dataset.columns[1:]:
    fig = plt.figure(figsize = (12,8))
    sns.boxplot(y = 'diagnosis', x = i, data = dataset)

    plt.title('Diagnosis vs '+i)
    plt.savefig('image/visual/Box plot/'+i+'.jpeg')

# %%
"""
Outlier correction using Quantile function.
quantile function -  return conditional median value.
we replace outliers with minimum and maximum margin values

"""

# %%
"""
Radius mean outlier correction
"""

# %%

#x=dfm['radius_mean'].quantile(0.03)
y=dfm['radius_mean'].quantile(0.98)
#dfm['radius_mean'] = np.where(dfm['radius_mean'] <x, x,dfm['radius_mean']) replacing lower outlier with conditional median value
dfm['radius_mean'] = np.where(dfm['radius_mean'] >y, y,dfm['radius_mean']) # replacing upper outlier with conditional median value

x=dfb['radius_mean'].quantile(0.01)
y=dfb['radius_mean'].quantile(0.996)
dfb['radius_mean'] = np.where(dfb['radius_mean'] <x, x,dfb['radius_mean'])
dfb['radius_mean'] = np.where(dfb['radius_mean'] >y, y,dfb['radius_mean'])


# %%
"""
Texture mean outlier correction
"""

# %%
x=dfm['texture_mean'].quantile(0.01)
y=dfm['texture_mean'].quantile(0.97)
dfm['texture_mean'] = np.where(dfm['texture_mean'] <x, x,dfm['texture_mean']) #
dfm['texture_mean'] = np.where(dfm['texture_mean'] >y, y,dfm['texture_mean'])

#x=dfb['texture_mean'].quantile(0.001)
y=dfb['texture_mean'].quantile(0.95)
#dfb['texture_mean'] = np.where(dfb['texture_mean'] <x, x,dfb['texture_mean'])
dfb['texture_mean'] = np.where(dfb['texture_mean'] >y, y,dfb['texture_mean'])

# %%
"""
Smoothness mean outlier correction
"""

# %%
#x=dfm['smoothness_mean'].quantile(0.00)
y=dfm['smoothness_mean'].quantile(0.98)
#dfm['smoothness_mean'] = np.where(dfm['smoothness_mean'] <x, x,dfm['smoothness_mean'])
dfm['smoothness_mean'] = np.where(dfm['smoothness_mean'] >y, y,dfm['smoothness_mean'])

x=dfb['smoothness_mean'].quantile(0.01)
y=dfb['smoothness_mean'].quantile(0.99)
dfb['smoothness_mean'] = np.where(dfb['smoothness_mean'] <x, x,dfb['smoothness_mean'])
dfb['smoothness_mean'] = np.where(dfb['smoothness_mean'] >y, y,dfb['smoothness_mean'])

# %%
"""
Compactness mean outlier correction
"""

# %%
#x=dfm['compactness_mean'].quantile(0.00)
y=dfm['compactness_mean'].quantile(0.9625)
#dfm['compactness_mean'] = np.where(dfm['compactness_mean'] <x, x,dfm['compactness_mean'])
dfm['compactness_mean'] = np.where(dfm['compactness_mean'] >y, y,dfm['compactness_mean'])

#x=dfb['compactness_mean'].quantile(0.0)
y=dfb['compactness_mean'].quantile(0.97)
#dfb['compactness_mean'] = np.where(dfb['compactness_mean'] <x, x,dfb['compactness_mean'])
dfb['compactness_mean'] = np.where(dfb['compactness_mean'] >y, y,dfb['compactness_mean'])


# %%
"""
Symmetry mean outlier correction
"""

# %%
#x=dfm['symmetry_mean'].quantile(0.03)
y=dfm['symmetry_mean'].quantile(0.975)
#dfm['symmetry_mean'] = np.where(dfm['symmetry_mean'] <x, x,dfm['symmetry_mean'])
dfm['symmetry_mean'] = np.where(dfm['symmetry_mean'] >y, y,dfm['symmetry_mean'])

x=dfb['symmetry_mean'].quantile(0.01)
y=dfb['symmetry_mean'].quantile(0.975)
dfb['symmetry_mean'] = np.where(dfb['symmetry_mean'] <x, x,dfb['symmetry_mean'])
dfb['symmetry_mean'] = np.where(dfb['symmetry_mean'] >y, y,dfb['symmetry_mean'])


# %%
"""
Fractal dimension mean outlier correction
"""

# %%
#x=dfm['fractal_dimension_mean'].quantile(0.03)
y=dfm['fractal_dimension_mean'].quantile(0.995)
#dfm['fractal_dimension_mean'] = np.where(dfm['fractal_dimension_mean'] <x, x,dfm['fractal_dimension_mean'])
dfm['fractal_dimension_mean'] = np.where(dfm['fractal_dimension_mean'] >y, y,dfm['fractal_dimension_mean'])

#x=dfb['fractal_dimension_mean'].quantile(0.01)
y=dfb['fractal_dimension_mean'].quantile(0.9525)
#dfb['fractal_dimension_mean'] = np.where(dfb['fractal_dimension_mean'] <x, x,dfb['fractal_dimension_mean'])
dfb['fractal_dimension_mean'] = np.where(dfb['fractal_dimension_mean'] >y, y,dfb['fractal_dimension_mean'])


# %%
"""
## Joining both M and B data
"""

# %%
dataset=pd.concat([dfm,dfb])

# %%
dataset # visualizing dataset

# %%
"""
###### visualizing corrected(without outliers) data with boxplot
"""

# %%
for i in dataset.columns[1:]:
    fig = plt.figure(figsize = (12,8))
    sns.boxplot(y = 'diagnosis', x = i, data = dataset)

    plt.title('Diagnosis vs '+i)
    plt.savefig('image/visual/box_plot_without_outlier/'+i+'.jpeg')

# %%
"""
# Importing Classification models
"""

# %%
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,accuracy_score,f1_score,roc_auc_score,roc_curve,plot_roc_curve,auc
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# %%
"""
## Feature Scaling
"""

# %%
x=dataset.drop('diagnosis',axis=1) #input dataset
x.shape

# %%
y=dataset['diagnosis']#output dataset
y.shape

# %%
"""
Change 'M' and 'B' to '1' and '0' respectively from diagnosis column 
"""

# %%
def change(n):
    if n=='M':
        return 1
    else:
        return 0
y=y.apply(change)

# %%
sd=StandardScaler() 
x=sd.fit_transform(x) #transforming data to standardized value
x.shape

# %%
"""
### Train test split
"""

# %%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40,random_state=32)#spliting train 80% and test 20%
acc_train=[]
acc=[]# accuracy
name=[] # model name
f1m=[] # f1 score M
f1b=[] # f1 score B
roc_auc=[]#roc_auc
cross_score=[]
# all the above list are used for model selection purpose
x.shape

# %%
y_train.value_counts() #getting count of M and B from train data

# %%
y_test.value_counts()  #getting count of M and B from test data
x.shape

# %%
"""
# Classification Models
"""

# %%
"""
## LogisticRegression
"""

# %%
lr = LogisticRegression(random_state=32)
lr.fit(x_train,y_train)# fitting the model
y_pred=lr.predict(x_test) #prediction based on x test



f1b.append(f1_score(y_test,y_pred,pos_label=1)) # append f1 score of B 
f1m.append(f1_score(y_test,y_pred,pos_label=0)) # append f1 score of M 
name.append('LogisticRegression')
acc.append(lr.score(x_test,y_test))#append accuracy of test data
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='icefire')
plt.savefig('image/selection/Heatmap/Logistic_Regression.png')
# visualize confusion matrix in heatmap



auc = roc_auc_score(y_test,lr.predict_proba(x_test)[:, 1])# getting roc curve by predict_proba method
plot_roc_curve(lr,x_test,y_test)
plt.savefig('image/selection/ROC_Curve/Logistic_Regression.png')
plt.show()
roc_auc.append(auc)

print('LogisticRegression')
print(classification_report(y_test,y_pred)) # getting classification report
print('Accuracy of Test: {:.2}'.format(lr.score(x_test,y_test)),end='\n\n')
print('Accuracy of Train: {:.2}'.format(lr.score(x_train,y_train)),end='\n\n')


# getting accuracy
print('F1_score of M',f1_score(y_test,y_pred,pos_label=0),end='\n\n') #getting f1 score for M
print('F1_score of B',f1_score(y_test,y_pred,pos_label=1),end='\n\n')#getting f1 score for M
print('AUC score',auc,end='\n\n')

acc_train.append(lr.score(x_train,y_train))#append accuracy of train data
#print(x.shape)
score=cross_val_score(lr,x,y,cv=5)# calculate cross validation score
#print(x.shape)
print('Logistic Regression cross Score \nmean: {}\nmin :{}\nmax :{}'.format(score.mean(),score.min(),score.max()))
cross_score.append(score.mean())#append the cross validation score 

# %%
"""
## KNeighborsClassifier
"""

# %%
knn=KNeighborsClassifier(n_neighbors=2)# setting n_neighbors 2 bacause we have 2 classification M & B
knn.fit(x_train,y_train)# fitting model
y_pred=knn.predict(x_test) #prediction based on x test

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='coolwarm')#heatmap for model
plt.show()

f1m.append(f1_score(y_test,y_pred,pos_label=0))

f1b.append(f1_score(y_test,y_pred,pos_label=1))
name.append('KNeighborsClassifier')#append model name
acc.append(knn.score(x_test,y_test))#append score
plt.savefig('image/selection/Heatmap/knn.png')

auc = roc_auc_score(y_test,knn.predict_proba(x_test)[:, 1])
plot_roc_curve(knn,x_test,y_test)
plt.savefig('image/selection/ROC_Curve/knn.png')
plt.show()
roc_auc.append(auc)

print('KNeighborsClassifier')
print(classification_report(y_test,y_pred)) #getting classification report
print('Accuracy of Test: {:.2}\n'.format(knn.score(x_test,y_test)))
print('Accuracy of Train: {:.2}\n'.format(knn.score(x_train,y_train)))
print('F1_score of M',f1_score(y_test,y_pred,pos_label=0),end='\n\n')#f1score
print('F1_score of B',f1_score(y_test,y_pred,pos_label=1),end='\n\n')#f1score
print('AUC score',auc,end='\n\n')


acc_train.append(knn.score(x_train,y_train))
score=cross_val_score(knn,x,y,cv=5)
print('Knn cross Score \nmean: {}\nmin :{}\nmax :{}'.format(score.mean(),score.min(),score.max()))
cross_score.append(score.mean())

# %%
"""
## Gaussian NB
"""

# %%
nb=GaussianNB()#model object
nb.fit(x_train,y_train)#fitting data
y_pred=nb.predict(x_test)#prediction based on x test

#print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='turbo',vmax=200)
plt.show()

f1m.append(f1_score(y_test,y_pred,pos_label=0))

f1b.append(f1_score(y_test,y_pred,pos_label=1))
name.append('Naive Bayes')
acc.append(nb.score(x_test,y_test))
plt.savefig('image/selection/Heatmap/Gaussian_NB.png')

auc = roc_auc_score(y_test,nb.predict_proba(x_test)[:, 1])
plot_roc_curve(nb,x_test,y_test)
plt.savefig('image/selection/ROC_Curve/Gaussian_NB.png')
plt.show()
roc_auc.append(auc)

print('Naive bayes')
print(classification_report(y_test,y_pred)) 
print('Accuracy of Test: {:.2}\n'.format(nb.score(x_test,y_test)))
print('Accuracy of Train: {:.2}\n'.format(nb.score(x_train,y_train)))
print('F1_score of M',f1_score(y_test,y_pred,pos_label=0),end='\n\n')
print('F1_score of B',f1_score(y_test,y_pred,pos_label=1),end='\n\n')
print('AUC score',auc,end='\n\n')

acc_train.append(nb.score(x_train,y_train))
score=cross_val_score(nb,x,y,cv=5)
print('Gaussiannb cross Score \nmean: {}\nmin :{}\nmax :{}'.format(score.mean(),score.min(),score.max()))
cross_score.append(score.mean())

# %%
"""
## Decision Tree Classifier
"""

# %%
dt=DecisionTreeClassifier(random_state=32)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

#print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='Greens',vmax=200)
plt.show()

name.append('Decision Tree')
f1m.append(f1_score(y_test,y_pred,pos_label=0))
f1b.append(f1_score(y_test,y_pred,pos_label=1))
acc.append(lr.score(x_test,y_test))
plt.savefig('image/selection/Heatmap/Desicion_tree.png')

auc = roc_auc_score(y_test,dt.predict_proba(x_test)[:, 1])
plot_roc_curve(dt,x_test,y_test)
plt.savefig('image/selection/ROC_Curve/Desicion_tree.png')
plt.show()
roc_auc.append(auc)


print('DecisionTreeClassifier')
print(classification_report(y_test,y_pred))
print('Accuracy of Test: {:.2}\n'.format(dt.score(x_test,y_test)))
print('Accuracy of Train: {:.2}\n'.format(dt.score(x_train,y_train)))
print('F1_score of M',f1_score(y_test,y_pred,pos_label=0),end='\n\n')
print('F1_score of B',f1_score(y_test,y_pred,pos_label=1),end='\n\n')
print('AUC score',auc,end='\n\n')


acc_train.append(dt.score(x_train,y_train))
score=cross_val_score(dt,x,y,cv=5)
print('Decision Tree \nmean: {}\nmin :{}\nmax :{}'.format(score.mean(),score.min(),score.max()))
cross_score.append(score.mean())

# %%
"""
## Random Forest Classification
"""

# %%
print(i)
rf=RandomForestClassifier(random_state=32) #setting n_estimator using accuracy
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)

#print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='ocean')
plt.show()

f1m.append(f1_score(y_test,y_pred,pos_label=0))

f1b.append(f1_score(y_test,y_pred,pos_label=1))
name.append('Random Forest')
acc.append(lr.score(x_test,y_test))
plt.savefig('image/selection/Heatmap/random_forest.png')


auc = roc_auc_score(y_test,dt.predict_proba(x_test)[:, 1])
plot_roc_curve(rf,x_test,y_test)
plt.savefig('image/selection/ROC_Curve/Random_forest.png')
plt.show()
roc_auc.append(auc)

print('RandomForestClassifier')
print(classification_report(y_test,y_pred))
print('Accuracy of Test: {:.2}\n'.format(rf.score(x_test,y_test)))
print('Accuracy of Train: {:.2}\n'.format(rf.score(x_train,y_train)))
print('F1_score of M',f1_score(y_test,y_pred,pos_label=0),end='\n\n')
print('F1_score of B',f1_score(y_test,y_pred,pos_label=1),end='\n\n')
print('AUC score',auc,end='\n\n')

acc_train.append(rf.score(x_train,y_train))
score=cross_val_score(rf,x,y,cv=5)
print('Random Forest \nmean: {}\nmin :{}\nmax :{}'.format(score.mean(),score.min(),score.max()))
cross_score.append(score.mean())

# %%
"""
## Support vector classification
"""

# %%
svc=SVC() #setting c and gamma value using searchgridCV
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)

#print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='plasma')
plt.show()

f1m.append(f1_score(y_test,y_pred,pos_label=0))

f1b.append(f1_score(y_test,y_pred,pos_label=1))
name.append('Support Vector')
acc.append(lr.score(x_test,y_test))
plt.savefig('image/selection/Heatmap/SVC.png')


auc = roc_auc_score(y_test,svc.decision_function(x_test))
plot_roc_curve(svc,x_test,y_test)
plt.savefig('image/selection/ROC_Curve/Support_vector.png')
plt.show()
roc_auc.append(0.5)



print('SVC')
print(classification_report(y_test,y_pred))
print('Accuracy of Test: {:.2}\n'.format(svc.score(x_test,y_test)))
print('Accuracy of Train: {:.2}\n'.format(svc.score(x_train,y_train)))
print('F1_score of M',f1_score(y_test,y_pred,pos_label=0),end='\n\n')
print('F1_score of B',f1_score(y_test,y_pred,pos_label=1),end='\n\n')
print('AUC score',auc,end='\n\n')

acc_train.append(svc.score(x_train,y_train))
score=cross_val_score(svc,x,y,cv=5)
print('\nSupport Vector Classification \nmean: {}\nmin :{}\nmax :{}'.format(score.mean(),score.min(),score.max()))
cross_score.append(score.mean())

# %%
"""
## Gradient Boosting
"""

# %%
gbc=GradientBoostingClassifier(random_state=32)
gbc.fit(x_train,y_train)
y_pred=gbc.predict(x_test)

#print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='tab10')
plt.show()

f1m.append(f1_score(y_test,y_pred,pos_label=0))

f1b.append(f1_score(y_test,y_pred,pos_label=1))
name.append('Gradient Boosting')
acc.append(lr.score(x_test,y_test))
plt.savefig('image/selection/Heatmap/Gradiant_boosting.png')

auc = roc_auc_score(y_test,gbc.predict_proba(x_test)[:, 1])
plot_roc_curve(gbc,x_test,y_test)
plt.savefig('image/selection/ROC_Curve/Gradient_boosting.png')
plt.show()
roc_auc.append(auc)

print('GradientBoostingClassifier')
print(classification_report(y_test,y_pred))
print('Accuracy of Test: {:.2}'.format(gbc.score(x_test,y_test)))
print('Accuracy of Train: {:.2}'.format(gbc.score(x_train,y_train)))
print('F1_score of M',f1_score(y_test,y_pred,pos_label=0))
print('F1_score of B',f1_score(y_test,y_pred,pos_label=1))
print('AUC score',auc)


acc_train.append(gbc.score(x_train,y_train))
print(x.shape)
score=cross_val_score(gbc,x,y,cv=5)
print(x.shape)
print('Gradient Boosting \nmean: {}\nmin :{}\nmax :{}'.format(score.mean(),score.min(),score.max()))
cross_score.append(score.mean())

# %%
"""
# Model Selection
"""

# %%
"""
#### Creating Dataframe for Model selection purpose
"""

# %%
dic={'Logistic':lr,'Kneighbour':knn,'Naive bayes':nb,'Desicion Tree':dt,'Random Forest':rf,'Gradeint Boosting':gbc}
plt.figure(figsize=(18,8))
var=[]
for i in dic.keys():
    y_pred=dic[i].predict_proba(x_test)
    fpr,tpr,threshold=roc_curve(y_test,y_pred[:,1])
    #print(i)
    plt.plot(fpr,tpr)
    var.append(i+' - {:.4}'.format(roc_auc_score(y_test,y_pred[:,1])))
#print(var)
plt.legend(labels=var)
plt.show()
plt.savefig('image/selection/ROC_Curve/cumulative_curve.jpeg')


# %%
dic={'Model':name,'Accuracy Test':acc,'M F1 score':f1m,'B F1 score':f1b,'AUC':roc_auc,"Cross_val_score":cross_score,'Accuracy Train':acc_train}
result=pd.DataFrame(dic)
result


import dataframe_image as img
img.export(result,'Model selection Table.jpeg')


# %%
"""
## Model Selection with Accuracy & F1 score
"""

# %%
"""
Plot the Accuracy for both test and train data,Cross validation,f1 score,Auc score(roc_curve) using bar plot
"""

# %%
import plotly.graph_objects as go
model=result['Model']
fig = go.Figure(go.Bar(x=model, y=result['Accuracy Test'], name='Accuracy Test'))
fig.add_trace(go.Bar(x=model, y=result['Cross_val_score'], name='Cross Score'))
fig.add_trace(go.Bar(x=model, y=result['Accuracy Train'], name='Accuracy Train'))
fig.add_trace(go.Bar(x=model, y=result['M F1 score'], name='F1 M score'))
fig.add_trace(go.Bar(x=model, y=result['B F1 score'], name='F1 B score'))
fig.add_trace(go.Bar(x=model, y=result['AUC'], name='AUC Score'))
fig.update_layout(barmode='overlay',yaxis_range=[0.8,1])
fig.update_xaxes()
fig.update_traces(opacity=0.4)
fig.show()
fig.write_image('image/selection/model_selection.png')


# %%
"""
From above graph, logistic and Gradient Boosting has above 0.92 score in AUC,F1 and Cross val scores and  so we taking above model for future scope.
And we leave Desicion tree,Kneighbors,Naive Bayes,SVC and random forest because of low AUC , cross and accuracy of test even they had high accuracy of train data,this may be leads to overfit.
"""

# %%
acc_train=[]
acc=[]# accuracy
name=[] # model name
f1m=[] # f1 score M
f1b=[] # f1 score B
roc_auc=[]
cross_score=[]

# %%
"""
## Tuning with Hyper Paramters
"""

# %%
"""
Tuning parameters using GridSearchCV and RepeatedStratifiedKFold
"""

# %%
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold

# %%
"""
# Logistic Tuning
"""

# %%
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
penalty =  ['none', 'l1', 'l2', 'elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]



grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x, y)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# %%
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
    #print("%f (%f) with: %r" % (mean, stdev, param))

# %%
lr = LogisticRegression(random_state=32,C=0.01,penalty='l2',solver='liblinear')
lr.fit(x_train,y_train)# fitting the model
y_pred=lr.predict(x_test) #prediction based on x test



f1b.append(f1_score(y_test,y_pred,pos_label=1)) # append f1 score of B to the model
f1m.append(f1_score(y_test,y_pred,pos_label=0)) # append f1 score of M to the model
name.append('LogisticRegression')
acc.append(lr.score(x_test,y_test))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='icefire')
plt.savefig('image/selection/Heatmap/Logistic_Regression.png')
# visualize confusion matrix in heatmap



auc = roc_auc_score(y_test,lr.predict_proba(x_test)[:, 1])# getting roc curve by predict_proba method
plot_roc_curve(lr,x_test,y_test)
plt.savefig('image/selection/ROC_Curve/Logistic_Regression.png')
plt.show()
roc_auc.append(auc)

print('LogisticRegression')
print(classification_report(y_test,y_pred)) # getting classification report
print('Accuracy of Test: {:.2}'.format(lr.score(x_test,y_test)),end='\n\n')
print('Accuracy of Train: {:.2}'.format(lr.score(x_train,y_train)),end='\n\n')


# getting accuracy
print('F1_score of M',f1_score(y_test,y_pred,pos_label=0),end='\n\n') #getting f1 score for M
print('F1_score of B',f1_score(y_test,y_pred,pos_label=1),end='\n\n')#getting f1 score for M
print('AUC score',auc,end='\n\n')

acc_train.append(lr.score(x_train,y_train))
#print(x.shape)
score=cross_val_score(lr,x,y,cv=5)
#print(x.shape)
print('Logistic Regression cross Score \nmean: {}\nmin :{}\nmax :{}'.format(score.mean(),score.min(),score.max()))
cross_score.append(score.mean())

# %%
"""
# Gradiant Boosting Tuning
"""

# %%
#model = GradientBoostingClassifier()
#n_estimators = [10, 100, 1000]
#learning_rate = [0.001, 0.01, 0.1]
#subsample = [0.5, 0.7, 1.0]
#max_depth = [3, 7, 9]


# %%
# define grid search
#grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
#grid_result = grid_search.fit(x, y)


# %%
# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
   # print("%f (%f) with: %r" % (mean, stdev, param))

# %%
gbc=GradientBoostingClassifier(random_state=32,learning_rate=0.01,max_depth=3,n_estimators=1000,subsample=1.0)
gbc.fit(x_train,y_train)
y_pred=gbc.predict(x_test)

#print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='tab10')
plt.show()

f1m.append(f1_score(y_test,y_pred,pos_label=0))

f1b.append(f1_score(y_test,y_pred,pos_label=1))
name.append('Gradient Boosting')
acc.append(lr.score(x_test,y_test))
plt.savefig('image/selection/Heatmap/Gradiant_boosting.png')

auc = roc_auc_score(y_test,gbc.predict_proba(x_test)[:, 1])
plot_roc_curve(gbc,x_test,y_test)
plt.savefig('image/selection/ROC_Curve/Gradient_boosting.png')
plt.show()
roc_auc.append(auc)

print('GradientBoostingClassifier')
print(classification_report(y_test,y_pred))
print('Accuracy of Test: {:.2}'.format(gbc.score(x_test,y_test)))
print('Accuracy of Train: {:.2}'.format(gbc.score(x_train,y_train)))
print('F1_score of M',f1_score(y_test,y_pred,pos_label=0))
print('F1_score of B',f1_score(y_test,y_pred,pos_label=1))
print('AUC score',auc)


acc_train.append(gbc.score(x_train,y_train))
print(x.shape)
score=cross_val_score(gbc,x,y,cv=5)
print(x.shape)
print('Gradient Boosting \nmean: {}\nmin :{}\nmax :{}'.format(score.mean(),score.min(),score.max()))
cross_score.append(score.mean())

# %%
"""
# Model selection after tuning
"""

# %%
dic={'Logistic':lr,'Gradeint Boosting':gbc}
plt.figure(figsize=(18,8))
var=[]
for i in dic.keys():
    y_pred=dic[i].predict_proba(x_test)
    fpr,tpr,threshold=roc_curve(y_test,y_pred[:,1])
    #print(i)
    plt.plot(fpr,tpr)
    var.append(i+' - {:.4}'.format(roc_auc_score(y_test,y_pred[:,1])))
#print(var)
plt.legend(labels=var)
plt.show()
plt.savefig('image/selection/ROC_Curve/cumulative_curve.jpeg')

# %%
dic={'Model':name,'Accuracy Test':acc,'M F1 score':f1m,'B F1 score':f1b,'AUC':roc_auc,"Cross_val_score":cross_score,'Accuracy Train':acc_train}
res=pd.DataFrame(dic)
res


import dataframe_image as img
img.export(result,'Model selection Table.jpeg')


# %%
com=pd.concat([result[result['Model']=='LogisticRegression'],res[res['Model']=='LogisticRegression'],result[result['Model']=='Gradient Boosting'],res[res['Model']=='Gradient Boosting']])
com.set_index([pd.Index(['Before','After',"Before",'After'])])

# %%
img.export(result,'Model Comparision.jpeg')

# %%
import plotly.graph_objects as go
model=res['Model']
fig = go.Figure(go.Bar(x=model, y=res['Accuracy Test'], name='Accuracy Test'))
fig.add_trace(go.Bar(x=model, y=res['Cross_val_score'], name='Cross Score'))
fig.add_trace(go.Bar(x=model, y=res['Accuracy Train'], name='Accuracy Train'))
fig.add_trace(go.Bar(x=model, y=result['M F1 score'], name='F1 M score'))
fig.add_trace(go.Bar(x=model, y=result['B F1 score'], name='F1 B score'))
fig.add_trace(go.Bar(x=model, y=res['AUC'], name='AUC Score'))
fig.update_layout(barmode='overlay',yaxis_range=[0.8,1])
fig.update_xaxes()
fig.update_traces(opacity=0.3)
fig.show()
fig.write_image('image/selection/model_selection_after_tuning.png')

# %%
"""
# Conclusion
"""

# %%
"""
From the above graph,
Gradient boosting has high Cross_validation and Accuracy of test data.So it is good model.If we take the accuracy of train data,it shows '1'.It may leads to overfit.
"""

# %%
"""
And per Area under curve score logistic beats gradient boosting
"""

# %%
"""
So We conculde that We can use Both Logistic and Gradient Boosting model.
"""

# %%
"""
#### saving model for future refernce
"""

# %%
import pickle

# Save tuple
pickle.dump(gbc, open("model/gradient_model.pkl", 'wb'))
pickle.dump(gbc, open("model/logistic_model.pkl", 'wb'))
# Restore tuple
gbc_model = pickle.load(open("model/logistic_model.pkl", 'rb'))
lr_model = pickle.load(open("model/gradient_model.pkl", 'rb'))

# %%

y_pred=lr_model.predict(sd.transform([[15,20,0.08,0.010006,0.17,0.0567]])) #predicting model for dynamic data
y_pred

# %%
y_pred=gbc_model.predict(sd.transform([[15,20,0.08,0.010006,0.17,0.0567]])) #predicting model for dynamic data
y_pred

# %%
"""
                                                                             By Muthu Periyal and Hariharan 
"""