import  pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
train_data=pd.read_csv('train.csv')
hitmapTemp=train_data[['Pclass','Age','SibSp','Parch','Fare','Survived']]
hitmapData=hitmapTemp.corr()
#dataframe.corr()计算列与列的相关系数，取值范围为[-1,1] 接近1时，表示两者具有强烈的正相关性
#接近-1时，表示有强烈的负相关性
# f,ax=plt.subplots(figsize=(12,12))
# sns.heatmap(hitmapData,vmax=1,square=True)
# plt.show()
# plt.savefig('Corrlation-Matrix.png')
# hitmap_dict=hitmapData['Survived'].to_dict()
# for ele in sorted(hitmap_dict.items(),key=lambda  x:-abs(x[1])):
#     print(ele)

a=train_data[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(a)

# plt.figure(figsize=(12,6))
# sns.boxplot(x='Pclass',y='Age',data=train_data)
# xt=plt.xticks(rotation=45)
# # plt.show()
#
# plt.figure(figsize=(12,6))
# sns.regplot(x='Age',y='Survived',data=train_data)
# plt.title('Age')
# # plt.show()
#
# plt.figure(figsize=(12,6))
# grid=sns.FacetGrid(train_data,row='Pclass',col='Sex',size=2.2,aspect=1.6)
# grid.map(plt.hist,'Age',bins=10)
# grid.add_legend()
# # plt.show()
#
#
# plt.figure(figsize=(12,6))
# grid=sns.FacetGrid(train_data,col='Pclass',size=2.2,aspect=1.6)
# grid.map(plt.hist,'Age',bins=10)
# grid.add_legend()
# plt.show()


# train_data['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
# train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
# train_data['Cabin']=train_data.Cabin.fillna('U0')

from sklearn.ensemble import RandomForestRegressor
age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull=age_df.loc[train_data['Age'].notnull()]
# print(age_df_notnull)
age_df_isnull=age_df.loc[(train_data['Age'].isnull())]
X=age_df_notnull.iloc[:,1:]
# print(X)
Y=age_df_notnull.iloc[:,0]
# print(Y)
RFR=RandomForestRegressor(n_estimators=1000,n_jobs=-1) #n_estimators 决策树的个数  bootstrap=True：是否有放回的采样  n_jobs并行job个数
RFR.fit(X,Y)
predictAges=RFR.predict(age_df_isnull.iloc[:,1:])
train_data.loc[train_data['Age'].isnull(),['Age']]=predictAges
# print(train_data['Age'].head(10))
# print(train_data.info())

# Survived_0=train_data.Pclass[train_data.Survived==0].value_counts()
# Survived_1=train_data.Pclass[train_data.Survived==1].value_counts()
# P_S=pd.DataFrame({'S_0':Survived_0,u'S_1':Survived_1})
# print(P_S)
# P_S.plot(kind='bar',stacked=True)
# plt.title('Pclass_Survived')
# plt.xlabel('Pclass')
# plt.ylabel('Survived')
# plt.show()

# train_data.loc[train_data['Sex'].values=='female','Sex']=0
# train_data.loc[train_data['Sex'].values=='male','Sex']=1
# print(train_data['Sex'].head(10))
# a=train_data[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
# print(a)
# a.plot(kind='bar',stacked=True)
# plt.show()

# bins=[0,10,18,30,60,80]
# train_data['Age_band']=pd.cut(train_data['Age'],bins)
# a=train_data[['Age_band','Survived']].groupby('Age_band',as_index=True).mean().sort_values(by='Survived',ascending=True)
# a.plot(kind='bar',stacked=True)#stacked 堆积
# plt.xticks(rotation=45)
# # plt.show()
# # print(a)
#
# train_data['Title']=train_data['Name'].str.extract('([A-Za-z]+)\.',expand=False)
# a=pd.crosstab(train_data['Title'],train_data['Sex'])
# # a.plot(kind='bar',stacked=True)
# ax=train_data[['Title','Survived']].groupby(['Title']).mean()
# ax.plot(kind='bar',stacked=True)
# # plt.show()
# # print(a)
#
# #有无兄弟姐妹与存活率的关系
# SibSp_S=train_data[train_data['SibSp']!=0]
# NoSibSp_S=train_data[train_data['SibSp']==0]
# plt.figure(figsize=(10,5))
# plt.subplot(121)
# ax=SibSp_S['Survived'].value_counts()
# ax.plot(kind='pie',labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
# plt.xlabel('sibsp')
#
# plt.subplot(122)
# ax1=NoSibSp_S['Survived'].value_counts()
# ax1.plot(kind='pie',labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
# plt.xlabel('no_sibsp')
# plt.show()


# train_data['Family_Size']=train_data['Parch']+train_data['SibSp']+1
# a=train_data[['Family_Size','Survived']].groupby(['Family_Size']).mean()
# a.plot(kind='bar')
# plt.show()

fare_not_survived = train_data['Fare'][train_data['Survived'] == 0]
fare_survived = train_data['Fare'][train_data['Survived'] == 1]

average_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])
average_fare.plot(yerr=std_fare, kind='bar', legend=False)

# plt.show()

train_data.loc[train_data.Cabin.isnull(),'Cabin']='U0'
train_data['Has_Cabin']=train_data['Cabin'].apply(lambda x:0 if x=='U0' else 1)
ax=train_data[['Has_Cabin','Survived']].groupby(['Has_Cabin']).mean()
ax.plot(kind='bar')
# plt.show()

train_data['CabinLetter']=train_data['Cabin'].map(lambda x:re.compile("([a-zA-Z]+)").search(x).group())
train_data['CabinLetter']=pd.factorize(train_data['CabinLetter'])[0]
ax=train_data[['CabinLetter','Survived']].groupby(['CabinLetter']).mean()
ax.plot(kind='bar')
# plt.show()
#处理后发现Cabin生存率差别不大，所以可以将该特征删除


train_data=train_data.drop(['Has_Cabin','CabinLetter'],axis=1)
# print(train_data.head(10))

sns.countplot('Embarked',hue='Survived',data=train_data)
plt.title('Embarked and Survived')

sns.factorplot('Embarked','Survived',data=train_data,size=3,aspect=2)
plt.title('Embarked and Survived rate')
# plt.show()

embark_dummies=pd.get_dummies(train_data['Embarked'])
train_data=train_data.join(embark_dummies)
train_data.drop(['Embarked'],axis=1,inplace=True)


assert np.size(train_data['Age'])==891

scaler=preprocessing.StandardScaler()
train_data['Age_scaled']=scaler.fit_transform(train_data['Age'].values.reshape(-1,1))


train_data['Fare_bin']=pd.qcut(train_data['Fare'],5)


train_data['Fare_bin_id']=pd.factorize(train_data['Fare_bin'])[0]
fare_bin_dummies=pd.get_dummies(train_data['Fare_bin']).rename(columns=lambda  x:'Fare_'+str(x))
train_data=pd.concat([train_data,fare_bin_dummies],axis=1)
# print(train_data.head(10))

train_data_org=pd.read_csv('train.csv')
test_data_org=pd.read_csv('test.csv')
test_data_org['Survived']=0
combined_train_test=train_data_org.append(test_data_org)
combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0],inplace=True)
combined_train_test['Embarked']=pd.factorize(combined_train_test['Embarked'])[0]
embarked_dummies_df=pd.get_dummies(combined_train_test['Embarked'],prefix='Embarked')
combined_train_test=pd.concat([combined_train_test,embarked_dummies_df],axis=1)

# print(combined_train_test['Sex'].head(10))
combined_train_test['Sex']=pd.factorize(combined_train_test['Sex'])[0]
sex_dummies_df=pd.get_dummies(combined_train_test['Sex'],prefix='Sex')
combined_train_test=pd.concat([combined_train_test,sex_dummies_df],axis=1)

# print(combined_train_test.Age.head(10))

combined_train_test['Title']=combined_train_test['Name'].map(lambda x:re.compile(",(.*?)\.").findall(x)[0])
# combined_train_test['Title']=pd.factorize(combined_train_test['Title'])[0]
# print(combined_train_test['Title'])

# print(combined_train_test['Title'].unique())

title_Dict={}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
combined_train_test['Title']=combined_train_test['Title'].map(title_Dict)

combined_train_test['Name_length']=combined_train_test['Name'].apply((len))
# print(combined_train_test['Name_length'])

# print(combined_train_test[combined_train_test.Fare.notnull()].iloc[:,:])
combined_train_test['Fare']=combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
combined_train_test['Fare_bin']=pd.qcut(combined_train_test['Fare'],5)
combined_train_test['Fare_bin_id']=pd.factorize(combined_train_test['Fare_bin'])[0]
fare_bin_dummies=pd.get_dummies(combined_train_test['Fare_bin_id'],prefix='Fare_id')
combined_train_test=pd.concat([combined_train_test,fare_bin_dummies],axis=1)
combined_train_test.drop(['Fare_bin'],axis=1,inplace=True)
# print(combined_train_test.head())

def pclass_fare_category(df,pclass1_mean_fare,pclass2_mean_fare,pclass3_mean_fare):
    if df['Pclass']==1:
        if df['Fare']<=pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass']==2:
        if df['Fare']<=pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'

Pclass1_mean_fare=combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get(1)
Pclass2_mean_fare=combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get(2)
Pclass3_mean_fare=combined_train_test['Fare'].groupby(by=combined_train_test['Pclass']).mean().get(3)

combined_train_test['Pclass_Fare_Category']=combined_train_test.apply(pclass_fare_category,args=(Pclass1_mean_fare,Pclass2_mean_fare,Pclass3_mean_fare),axis=1)
pclass_level=LabelEncoder()
pclass_level.fit(np.array(['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))
combined_train_test['Pclass_Fare_Category']=pclass_level.transform(combined_train_test['Pclass_Fare_Category'])
pclass_dummies_df=pd.get_dummies(combined_train_test['Pclass_Fare_Category']).rename(columns=lambda x:'Pclass_'+str(x))
combined_train_test=pd.concat([combined_train_test,pclass_dummies_df])
combined_train_test['Pclass']=pd.factorize(combined_train_test['Pclass'])[0] #因式分解 的意思，将离散值映射为数字并唯一化。
# print(combined_train_test.columns)

def family_size_category(family_size):
    if family_size<=1:
        return 'Single'
    elif family_size<=4:
        return 'Small_Family'
    else:
        return 'Large_Family'

combined_train_test['Family_Size']=combined_train_test['Parch']+combined_train_test['SibSp']+1
combined_train_test['Family_Size_Category']=combined_train_test['Family_Size'].map(family_size_category)
# print(combined_train_test['Family_Size_Category'])
le_family=LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category']=le_family.transform(combined_train_test['Family_Size_Category'])
family_size_dummies_df=pd.get_dummies(combined_train_test['Family_Size_Category'],prefix=combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test=pd.concat([combined_train_test,family_size_dummies_df],axis=1)
# print(combined_train_test)
missing_age_df=pd.DataFrame(combined_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass']])
missing_age_train=missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test=missing_age_df[missing_age_df['Age'].isnull()]
print(missing_age_test.head(10))

from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


# def fill_missing_age(missing_age_train, missing_age_test):
#     missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
#     missing_age_Y_train = missing_age_train['Age']
#     missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
#
#     # model 1  gbm
#     gbm_reg = GradientBoostingRegressor(random_state=42)
#     gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
#     gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1,
#                                                 scoring='neg_mean_squared_error')
#     gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
#     print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
#     print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
#     print('GB Train Error for "Age" Feature Regressor:' + str(
#         gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
#     missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
#     print(missing_age_test['Age_GB'][:4])
#     # model 2 rf
#     rf_reg = RandomForestRegressor()
#     rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
#     rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
#                                                scoring='neg_mean_squared_error')
#     rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
#     print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
#     print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
#     print(
#         'RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
#     missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
#     print(missing_age_test['Age_RF'][:4])
#     # two models merge
#     print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
#     # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)
#     missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
#     print(missing_age_test['Age'][:4])
#     missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)
#     return missing_age_test
# combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)
#
# combined_train_test['Ticket_Letter']=combined_train_test['Ticket'].str.split().str[0]
# combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)
# combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]

combined_train_test.loc[combined_train_test.Cabin.isnull(),'Cabin']='U0'
combined_train_test['Cabin']=combined_train_test['Cabin'].apply(lambda x:0 if x=='U0' else 1)
Correlation = pd.DataFrame(combined_train_test[
 ['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass',
  'Pclass_Fare_Category', 'Age', 'Cabin']])
# colormap=plt.cm.viridis
# plt.figure(figsize=(12,4))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True, cmap=colormap, linecolor='white', annot=True)



g = sns.pairplot(combined_train_test[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
    u'Family_Size', u'Title', ]], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
# plt.show()


scale_age_fare=preprocessing.StandardScaler().fit(combined_train_test[['Age','Fare', 'Name_length']])
combined_train_test[['Age','Fare', 'Name_length']]=scale_age_fare.transform(combined_train_test[['Age','Fare', 'Name_length']])


#模型融合以及测试
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    # random forest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:10]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Feature from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best ET Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # merge the three models
    features_top_n = pd.concat(
        [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
        ignore_index=True).drop_duplicates()

    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                     feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)

    return features_top_n, features_importance

feature_to_pick = 30
feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])

from sklearn.model_selection import KFold

# Some useful parameters which will come in handy later on
ntrain = titanic_train_data_X.shape[0]
ntest = titanic_test_data_X.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 7 # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6,
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)

ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)

et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)

dt = DecisionTreeClassifier(max_depth=8)

knn = KNeighborsClassifier(n_neighbors = 2)

svm = SVC(kernel='linear', C=0.025)


#将pandas转换为arrays：
x_train = titanic_train_data_X.values # Creates an array of the train data
x_test = titanic_test_data_X.values # Creats an array of the test data
y_train = titanic_train_data_Y.values

rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test) # AdaBoost
et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) # Extra Trees
gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) # Gradient Boost
dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test) # Decision Tree
knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test) # KNeighbors
svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test) # Support Vector

print("Training is complete")

x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

from xgboost import XGBClassifier
gbm=XGBClassifier(n_estimators=2000,max_depth=4,min_child_weight=2,gamma=0.9,subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')



#构建学习曲线
from sklearn.learning_curve import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

X = x_train
Y = y_train

# RandomForest
rf_parameters = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2,
              'max_features' : 'sqrt','verbose': 0}

# AdaBoost
ada_parameters = {'n_estimators':500, 'learning_rate':0.1}

# ExtraTrees
et_parameters = {'n_jobs': -1, 'n_estimators':500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}

# GradientBoosting
gb_parameters = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}

# DecisionTree
dt_parameters = {'max_depth':8}

# KNeighbors
knn_parameters = {'n_neighbors':2}

# SVM
svm_parameters = {'kernel':'linear', 'C':0.025}

# XGB
gbm_parameters = {'n_estimators': 2000, 'max_depth': 4, 'min_child_weight': 2, 'gamma':0.9, 'subsample':0.8,
               'colsample_bytree':0.8, 'objective': 'binary:logistic', 'nthread':-1, 'scale_pos_weight':1}

title = "Learning Curves"
plot_learning_curve(RandomForestClassifier(**rf_parameters), title, X, Y, cv=None,  n_jobs=4, train_sizes=[50, 100, 150, 200, 250, 350, 400, 450, 500])
plt.show()