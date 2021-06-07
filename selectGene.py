from sklearn.preprocessing import MinMaxScaler
# preprocessing by mic
from sklearn.feature_selection import SelectKBest
import numpy as np
from minepy import MINE
def premic(X, Y):
    '''
    preprocess: select 60% of all features
    
    return:the index of selected features 
    '''
    num = round(X.shape[1] * 0.6)
    def mic(x, Y):
        mine = MINE()
        mine.compute_score(x,Y)
        return mine.mic()
    selector = SelectKBest(lambda X,Y: np.array(list(map(lambda x: mic(x,Y),X.T))), k=num)
    selector.fit_transform(X, Y)
    return (selector.get_support(indices=True))

from sklearn.ensemble import RandomForestRegressor
def rf_handler(X, Y, feature_names):
    '''
    select the best 50 features by random forest
    
    return: selected features
    '''
    rf = RandomForestRegressor()
    rf.fit(X,Y)
    importance = rf.feature_importances_.reshape(-1,1)
    mm = MinMaxScaler()
    mm.fit(importance)
    result = list(zip(feature_names, importance.reshape(-1)))
    result.sort(key=lambda a:a[1], reverse=True)
    return result[:50]

from sklearn.linear_model import LinearRegression
def lr_handler(X, Y, feature_names):
    '''
    select the best 50 features by linear regression
    
    return: selected features
    '''
    lr = LinearRegression()
    lr.fit(X,Y)
    importance = np.abs(lr.coef_).reshape(-1,1)
    mm = MinMaxScaler()
    mm.fit(importance)
    result = list(zip(feature_names, importance.reshape(-1)))
    result.sort(key=lambda a:a[1], reverse=True)
    return result[:50]

from sklearn.linear_model import Lasso
def lasso_handler(X, Y, feature_names):
    '''
    select the best 50 features by Lasso

    return: selected features
    '''
    lasso = Lasso(alpha=.01)
    lasso.fit(X,Y)
    importance = np.abs(lasso.coef_).reshape(-1,1)
    mm = MinMaxScaler()
    mm.fit(importance)
    result = list(zip(feature_names, importance.reshape(-1)))
    result.sort(key=lambda a:a[1], reverse=True)
    return result[:50]

from sklearn.linear_model import Ridge
def ridge_handler(X, Y, feature_names):
    '''
    select the best 50 features by Ridge

    return: selected features
    '''
    ridge = Ridge(alpha=10)
    ridge.fit(X,Y)
    importance = np.abs(ridge.coef_).reshape(-1,1)
    mm = MinMaxScaler()
    mm.fit(importance)
    result = list(zip(feature_names, importance.reshape(-1)))
    result.sort(key=lambda a:a[1], reverse=True)
    return result[:50]

from stability_selection.randomized_lasso import RandomizedLasso
def stab_handler(X, Y, feature_names):
    '''
    select the best 50 features by stability selection

    return: selected features
    '''
    rlasso = RandomizedLasso(alpha=0.025)
    rlasso.fit(X,Y)
    importance = np.abs(rlasso.coef_).reshape(-1,1)
    mm = MinMaxScaler()
    mm.fit(importance)
    result = list(zip(feature_names, importance.reshape(-1)))
    result.sort(key=lambda a:a[1], reverse=True)
    return result[:50]

from sklearn.feature_selection import RFE
def rfe_handler(X, Y, feature_names):
    '''
    select the best 50 features by recursive feature elimination

    return: selected features
    '''
    rfe = RFE(LinearRegression(), n_features_to_select=50)
    rfe.fit(X,Y)
    importance = np.abs(rfe.ranking_).reshape(-1,1)
    mm = MinMaxScaler()
    mm.fit(importance)
    result = list(zip(feature_names, importance.reshape(-1)))
    result.sort(key=lambda a:a[1], reverse=True)
    return result[:50]

from sklearn.ensemble import ExtraTreesClassifier
def dt_handler(X, Y, feature_names):
    '''
    select the best 50 features by decision tree

    return: selected features
    '''
    dt = ExtraTreesClassifier()
    dt.fit(X,Y)
    importance = np.abs(dt.feature_importances_).reshape(-1,1)
    mm = MinMaxScaler()
    mm.fit(importance)
    result = list(zip(feature_names, importance.reshape(-1)))
    result.sort(key=lambda a:a[1], reverse=True)
    return result[:50]

def final_handler(X, Y, feature_names):
    result_rf = rf_handler(X, Y, feature_names)
    result_lasso = lasso_handler(X, Y, feature_names)
    result_ridge = ridge_handler(X, Y, feature_names)
    result_stab = stab_handler(X, Y, feature_names)
    result_rfe = rfe_handler(X, Y, feature_names)
    result_dt = dt_handler(X, Y, feature_names)
    result_temp = result_rf+result_lasso+result_ridge+result_stab+result_rfe+result_dt
    result = [x[0] for x in result_temp]
    result = pd.value_counts(result)
    result_final = list(zip(result.keys().tolist(),result.tolist()))
    return result_final

# get data
import pandas as pd
import datetime
from sklearn import preprocessing
data = pd.read_excel('DLBCL.xlsx')
feature_names = data.columns[1:].values
X = data.iloc[0:,1:].values
Y = data.iloc[:,0].values
Y = preprocessing.LabelEncoder().fit_transform(Y)
startTime = datetime.datetime.now()
selected = premic(X, Y)
endTime = datetime.datetime.now()
print("time of preprocess:", endTime-startTime)
feature_names = feature_names[selected]
X = X[:,selected]

# select features by random forest
startTime = datetime.datetime.now()
result = rf_handler(X, Y, feature_names)
endTime = datetime.datetime.now()
print("time of random forest:", endTime-startTime)
print(result)

# select features by linear regression
startTime = datetime.datetime.now()
result = lr_handler(X, Y, feature_names)
endTime = datetime.datetime.now()
print("time of linear regression:", endTime-startTime)
print(result)

# select features by lasso
startTime = datetime.datetime.now()
result = lasso_handler(X, Y, feature_names)
endTime = datetime.datetime.now()
print("time of lasso:", endTime-startTime)
print(result)

# select features by ridge
startTime = datetime.datetime.now()
result = ridge_handler(X, Y, feature_names)
endTime = datetime.datetime.now()
print("time of ridge regression:", endTime-startTime)
print(result)

# select features by stability selection
startTime = datetime.datetime.now()
result = stab_handler(X, Y, feature_names)
endTime = datetime.datetime.now()
print("time of stability selection:", endTime-startTime)
print(result)

# select features by RFE
startTime = datetime.datetime.now()
result = rfe_handler(X, Y, feature_names)
endTime = datetime.datetime.now()
print("time of RFE:", endTime-startTime)
print(result)

# select features by decision tree
startTime = datetime.datetime.now()
result = dt_handler(X, Y, feature_names)
endTime = datetime.datetime.now()
print("time of decision tree:", endTime-startTime)
print(result)

startTime = datetime.datetime.now()
result_final = final_handler(X, Y, feature_names)
endTime = datetime.datetime.now()
print("time of final:", endTime-startTime)
print(result_final)