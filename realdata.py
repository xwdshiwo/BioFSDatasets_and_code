# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn.neural_network as nn
from sklearn import preprocessing
from sklearn.cluster import KMeans
from stability_selection import RandomizedLasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from minepy import MINE
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler


def read_file(fileName):
    '''
    read file

    param fileName: XXX.xlsx 
    return: dataset_name, X, Y, feature_names
    '''
    dataset_name = fileName.split('.')[0]
    data = pd.read_csv(fileName, header=None, sep='\t', low_memory=False).T
    feature_names = data.iloc[0, 1:].values
    X = data.iloc[1: , 1:129].apply(pd.to_numeric, axis=0).values
    Y = preprocessing.LabelEncoder().fit_transform(data.iloc[1:, 0].values)
    # If there is a null value in X, delete the column with null value
    if np.isnan(X).any():
        # Locate the null value in X
        nanrow, nancol = np.where(np.isnan(X))
        nancol = np.unique(nancol)
        print("null value：", nancol)
        # Delete columns with null values
        X = np.delete(X, nancol, axis=1)
        feature_names = np.delete(feature_names, nancol)

    # threshold = 0.85
    # k = 10
    # X, feature_names = kmeans(X, feature_names, k, threshold)
    # X, feature_names = preTtest(X, Y, feature_names, threshold)
    # X, feature_names = premic(X, Y, feature_names, threshold)
    return dataset_name, X, Y, feature_names


def file_info(X, Y):
    '''
    get information of dataset

    return: Number of positive samples, negative samples, total samples and features
    '''
    total_num = len(Y)
    pos_num = np.sum((Y == 1).astype(int))
    neg_num = np.sum((Y == 0).astype(int))
    feature_num = X.shape[1]
    return pos_num, neg_num, total_num, feature_num

# preprocess
def kmeans(X, feature_names, k, threshold):
    '''
    The feature is removed by KMeans

    K:表示聚类的簇的数量
    threshold:表示保留特征数量的阈值
    :return: Features after removing redundancy
    '''
    print("KM之前：", X.shape)
    scaler = StandardScaler()
    km_x = scaler.fit_transform(X)
    # 在稳定性特征选择之前先进行去冗余操作
    num_clusters = k
    # 筛选冗余特征的阈值，0.15表示删除每个簇中除种子节点外的前15%的特征
    my_threshold = 1 - threshold
    clusters_name = [[] for i in range(num_clusters)]
    clusters_index = [[] for i in range(num_clusters)]
    model_km = KMeans(n_clusters=num_clusters)
    km_result = list(model_km.fit_predict(km_x.T))
    features = list(zip(feature_names, km_result, list(range(km_x.shape[1]))))
    for feature in features:
        clusters_name[feature[1]].append(feature[0])
        clusters_index[feature[1]].append(feature[2])
    # print("聚类结果：",km_result,cluster_features)
    # print(type(clusters_name[0]), clusters_name)
    # print(type(clusters_index[0]), clusters_index)

    # 根据聚类结果，每个簇中index最小的被定义为簇的种子节点
    cluster_seeds = [min(cluster) for cluster in clusters_index]
    print("簇中种子点的index：", cluster_seeds)
    # print("X的大小", X.shape[0], X.shape[1])
    # 计算簇中每个点与种子点的相关系数
    node_scores = [[] for i in range(num_clusters)]
    for i in range(num_clusters):
        node_num = len(clusters_name[i])
        seed_feature = X[:, cluster_seeds[i]]
        j = 0
        for j in clusters_index[i]:
            index = j
            score = np.corrcoef(X[:, index], seed_feature.T)
            # 记录协方差
            node_scores[i].append(np.abs(score[0, 1]))
        temp_cluster_scores = list(zip(clusters_index[i], clusters_name[i], node_scores[i]))
        temp_cluster_scores.sort(key=lambda a: a[2], reverse=True)
        # print("删除前：",temp_cluster_scores)
        # 删除簇中除种子节点外的前15%的节点
        if node_num > 1.0 / my_threshold:
            # print("删除的节点：", temp_cluster_scores[1:int(node_num * my_threshold + 0.5) + 1])
            del temp_cluster_scores[1:int(node_num * my_threshold + 0.5) + 1]
        # print("删除后的：",temp_cluster_scores)
        clusters_index[i] = [x[0] for x in temp_cluster_scores]
        clusters_name[i] = [x[1] for x in temp_cluster_scores]

    result_index = []
    result_name = []
    for l in clusters_index:
        result_index += [i for i in l]
    for n in clusters_name:
        result_name += n
    result_x = X[:, result_index]
    print("KM之后的：", result_x.shape)
    return result_x, result_name

# 根据互信息进行预处理
def premic(X, Y, feature_names, threshold):
    ###设置数目
    num = round(X.shape[1] * threshold)
    print("mic预处理之后的数量：", num)
    # num = 1000
    mine = MINE()
    mic_scores = []
    for i in range(X.shape[1]):
        # i = i+1
        mine.compute_score(X[:,i], Y)
        m = mine.mic()
        mic_scores.append(m)
    mic_result = dict(zip(feature_names, mic_scores))
    mic_result_df = pd.DataFrame([mic_result]).T
    mic_sorted = mic_result_df.sort_values(axis=0, by=[0], ascending=False)
    mic_result = mic_sorted.iloc[0:num, :]

    temp_index = []
    # print(type(np.array(mic_result.index)),type(feature_names), np.array(mic_result.index))
    for name in mic_result.index:
        # print(name,X[:,list(feature_names).index(name)])
        temp_index.append(list(feature_names).index(name))
    minc_X = X[:,temp_index]
    return minc_X, np.array(mic_result.index)

# 根据t检验的值继续筛选
def preTtest(X, Y, feature_names, threshold):
    '''
    利用独立样本t检验来对样本进行排序

    :return: X，feature_names
    '''
    print(X.shape, Y.shape)
    # 按行进行拼接,然后对Y进行排序，方便进行正负样本对应基因值的T检验
    X_Y = pd.DataFrame(np.concatenate((Y.reshape([-1,1]), X), axis=1)).sort_values(by=0)
    #　print(type(X_Y), X_Y)
    ttest_x = X_Y.iloc[0:, 1:].apply(pd.to_numeric, axis=0).values
    ttest_y = X_Y.iloc[0:, 0].values
    pos_num = np.sum(ttest_y == 0)
    print("正样本",pos_num)
    temp_x = np.split(ttest_x,[pos_num])
    pos_X = temp_x[0]
    neg_X = temp_x[1]
    feature_p = []
    # 记录P值小于0.05的数量
    count = 0
    for j in range(ttest_x.shape[1]):
        t,p = ttest_ind(pos_X[:,j], neg_X[:,j])
        # (j,feature_names[j],t,p)
        feature_p.append(p)
        if p < 0.05: count += 1
    temp_features = list(zip(ttest_x.T, feature_names, feature_p))
    for i in list(zip(feature_names, feature_p)):
        print(i)
    temp_features.sort(key=lambda a:a[2], reverse=False)
    filter_features = temp_features[0: round(X.shape[1] * threshold)]
    ttest_X = np.array([a[0] for a in filter_features]).T
    ttest_names = np.array([a[1] for a in filter_features])
    print(count, "筛选之后的基因：", ttest_names)
    # print(new_X[:,0:30])
    return ttest_X, ttest_names

# stab
def stab_handler(X, Y, feature_names, num):
    '''
    select the best num features by stability selection

    return: selected x and features
    '''
    print("stab的大小：",X.shape)
    rlasso = RandomizedLasso(alpha=0.0001, max_iter=-1)
    rlasso.fit(X, Y)
    importance = np.abs(rlasso.coef_)
    # mm = MinMaxScaler()
    # mm.fit(importance)
    result = list(zip(feature_names, importance))
    result.sort(key=lambda a: a[1], reverse=True)
    # print("stab:", result)
    stab_result = result[:num]
    # Select the corresponding data information according to the selected protein name
    temp = []
    selected_names = []
    selected_weights = []
    for temp_feature in stab_result:
        temp.append(list(feature_names).index(temp_feature[0]))
        selected_names.append(temp_feature[0])
        selected_weights.append(temp_feature[1])
    x = X[:, temp]
    selected_weights = np.array(selected_weights).reshape(-1, 1)
    mm = MinMaxScaler()
    selected_weights = mm.fit_transform(selected_weights)
    return x, selected_names, selected_weights.reshape(-1)


# Lasso
def lasso_handler(X, Y, feature_names, num):
    '''
    select the best num features by Lasso

    return: selected x and features
    '''
    lasso = Lasso(alpha=1e-10, max_iter=-1)
    lasso.fit(X, Y)
    importance = np.abs(lasso.coef_)
    result = list(zip(feature_names, importance))
    result.sort(key=lambda a: a[1], reverse=True)
    lasso_result = result[:num]
    temp = []
    selected_names = []
    selected_weights = []
    for temp_feature in lasso_result:
        temp.append(list(feature_names).index(temp_feature[0]))
        selected_names.append(temp_feature[0])
        selected_weights.append(temp_feature[1])
    x = X[:, temp]
    selected_weights = np.array(selected_weights).reshape(-1, 1)
    mm = MinMaxScaler()
    selected_weights = mm.fit_transform(selected_weights)
    return x, selected_names, selected_weights.reshape(-1)


# Calculate the score of the model
def model_score(model, X, Y):
    model_accs = []

    # 留一法
    loo = LeaveOneOut()
    scores = 0
    for train_index, test_index in loo.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # print("TRAIN:", train_index, "TEST:", test_index)
        model.fit(X_train, y_train)
        model_accs.append(model.score(X_test, y_test))

    return np.array(model_accs)


# Lasso + SVM
def lasso_svm(X, Y, feature_names, num):
    # select features by lasso
    lasso_X, lasso_feature_names, feature_weights = lasso_handler(X, Y, feature_names, num)
    model_svm = svm.SVC(max_iter=-1, kernel='rbf')
    model_scores = model_score(model_svm, lasso_X, Y)
    return lasso_feature_names, feature_weights, model_scores


# Lasso + NN
def lasso_nn(X, Y, feature_names, num):
    # select features by lasso
    lasso_X, lasso_feature_names, feature_weights = lasso_handler(X, Y, feature_names, num)
    model_nn = nn.MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, learning_rate='adaptive',
                                learning_rate_init=0.001, max_iter=1000)
    model_scores = model_score(model_nn, lasso_X, Y)
    return lasso_feature_names, feature_weights, model_scores


# Stab + SVM
def stab_svm(X, Y, feature_names, num):
    # select features by stab
    stab_X, stab_feature_names, feature_weights = stab_handler(X, Y, feature_names, num)
    model_svm = svm.SVC(max_iter=-1, kernel='rbf')
    model_scores = model_score(model_svm, stab_X, Y)
    return stab_feature_names, feature_weights, model_scores


# Stab + NN
def stab_nn(X, Y, feature_names, num):
    # select features by stab
    stab_X, stab_feature_names, feature_weights = stab_handler(X, Y, feature_names, num)
    model_nn = nn.MLPClassifier(activation='tanh', solver='adam', alpha=0.0001, learning_rate='adaptive',
                                learning_rate_init=0.001, max_iter=1000)
    model_scores = model_score(model_nn, stab_X, Y)
    return stab_feature_names, feature_weights, model_scores


# feature count
def featrue_count(selected_result):
    dict = {}
    for f in selected_result:
        dict[f[0]] = dict.get(f[0], 0) + f[1]
    return dict




fileName = 'real.txt'
dataset_name, X, Y, feature_names = read_file(fileName)
print("文件（数据）名称：%s" % dataset_name)
pos_num, neg_num, total_num, feature_num = file_info(X, Y)
print("正样本数：%s, 负样本数：%s, 总样本数：%d, 特征数：%d" % (pos_num, neg_num, total_num, feature_num))

writer = pd.ExcelWriter(r'realResult\kmeans\result_kmeans.xlsx')
filename = r'realResult\\kmeans\\' + dataset_name + '_kmeans.txt'  ####################################################文件位置
print("文件名：", filename)
f = open(filename, mode='w', encoding='utf8')  #################################
f.write('当前数据集为:' + dataset_name + '\n*******************************************************\n')

# The number of features selected is 10
num = 15
# 定义稳定性选择重复的次数
stab_num = 100
# methonds
lasso_svm_feature_names, lasso_svm_feature_weights, lasso_svm_model_scores = lasso_svm(X, Y, feature_names, num)
print("方法名：Lasso+SVM\n选择的特征：%s\n特征的权重：%s\n平均准确率：%.4f\t%s\n"
      %(lasso_svm_feature_names, lasso_svm_feature_weights, lasso_svm_model_scores.mean(), lasso_svm_model_scores))
f.write("方法名：Lasso+SVM\n选择的特征：%s\n特征的权重：%s\n平均准确率：%.4f\t%s\n\n"
      %(lasso_svm_feature_names, lasso_svm_feature_weights, lasso_svm_model_scores.mean(), lasso_svm_model_scores))
lasso_svm_dict = featrue_count(list(zip(lasso_svm_feature_names, lasso_svm_feature_weights )))
lasso_svm_df = pd.DataFrame.from_dict(lasso_svm_dict, orient='index', columns=['weight'])
lasso_svm_df = lasso_svm_df.reset_index().rename(columns={'index':'lasso+svm'})
lasso_svm_df.to_excel(excel_writer=writer, sheet_name='Lasso+SVM', index=False)
# print(lasso_svm_df)

lasso_nn_feature_names, lasso_nn_feature_weights, lasso_nn_model_scores = lasso_nn(X, Y, feature_names, num)
print("方法名：Lasso+NN\n选择的特征：%s\n特征的权重：%s\n平均准确率：%.4f\t%s\n"
      %(lasso_nn_feature_names, lasso_nn_feature_weights, lasso_nn_model_scores.mean(), lasso_nn_model_scores))
f.write("方法名：Lasso+NN\n选择的特征：%s\n特征的权重：%s\n平均准确率：%.4f\t%s\n\n"
      %(lasso_nn_feature_names, lasso_nn_feature_weights, lasso_nn_model_scores.mean(), lasso_nn_model_scores))
lasso_nn_dict = featrue_count(list(zip(lasso_nn_feature_names, lasso_nn_feature_weights )))
lasso_nn_df = pd.DataFrame.from_dict(lasso_nn_dict, orient='index', columns=['weight'])
lasso_nn_df = lasso_nn_df.reset_index().rename(columns={'index':'lasso+nn'})
lasso_nn_df.to_excel(excel_writer=writer, sheet_name='Lasso+NN', index=False)

stab_svm_names = []
stab_svm_weight = []
stab_svm_count = 0
f.write("***************************stab+SVM******************************\n")
for i in range(stab_num):
    threshold = 0.842
    k = 8
    km_x, km_names = kmeans(X, feature_names, k, threshold)
    # X, feature_names = preTtest(X, Y, feature_names, threshold)
    # X, feature_names = premic(X, Y, feature_names, threshold)
    stab_svm_feature_names, stab_svm_feature_weights, stab_svm_model_scores = stab_svm(km_x, Y, km_names, num)
    print("方法名：Stab_SVM\n选择的特征：%s\n特征的权重：%s\n平均准确率：%.4f\t%s\n"
          %(stab_svm_feature_names, stab_svm_feature_weights, stab_svm_model_scores.mean(), stab_svm_model_scores))
    f.write("方法名：Stab_SVM\n选择的特征：%s\n特征的权重：%s\n平均准确率：%.4f\t%s\n\n"
          %(stab_svm_feature_names, stab_svm_feature_weights, stab_svm_model_scores.mean(), stab_svm_model_scores))
    if np.sum(stab_svm_model_scores) == 6:
        stab_svm_count += 1
        stab_svm_names += stab_svm_feature_names
        stab_svm_weight += stab_svm_feature_weights.tolist()
f.write("方法Stab+SVM全中的次数为：%d\n" % stab_svm_count)
# 计数次数
stab_svm_count_df = pd.DataFrame(pd.value_counts(stab_svm_names),columns=['count'])
stab_svm_count_df = stab_svm_count_df.reset_index().rename(columns={'index':'id'})
stab_svm_count_df.to_excel(excel_writer=writer, sheet_name='Stab+SVM_count', index=False)
# 计数权重
stab_svm_dict = featrue_count(list(zip(stab_svm_names, stab_svm_weight )))
stab_svm_weight_df = pd.DataFrame.from_dict(stab_svm_dict, orient='index', columns=['weight'])
stab_svm_weight_df = stab_svm_weight_df.reset_index().rename(columns={'index':'id'})
stab_svm_weight_df = stab_svm_weight_df.sort_values(by='weight', ascending=False)
stab_svm_weight_df.to_excel(excel_writer=writer, sheet_name='Stab+SVM_weight', index=False)


stab_nn_names = []
stab_nn_weight = []
stab_nn_count = 0
f.write("***************************stab+NN******************************\n")
for i in range(stab_num):
    threshold = 0.842
    k = 8
    km_x, km_names = kmeans(X, feature_names, k, threshold)
    stab_nn_feature_names, stab_nn_feature_weights, stab_nn_model_scores = stab_nn(km_x, Y, km_names, num)
    print("方法名：Stab+NN\n选择的特征：%s\n特征的权重：%s\n平均准确率：%.4f\t%s\n"
          % (stab_nn_feature_names, stab_nn_feature_weights, stab_nn_model_scores.mean(), stab_nn_model_scores))
    f.write("方法名：Stab+NN\n选择的特征：%s\n特征的权重：%s\n平均准确率：%.4f\t%s\n\n"
            % (stab_nn_feature_names, stab_nn_feature_weights, stab_nn_model_scores.mean(), stab_nn_model_scores))
    if np.sum(stab_nn_model_scores) == 6:
        stab_nn_count += 1
        stab_nn_names += stab_nn_feature_names
        stab_nn_weight += stab_nn_feature_weights.tolist()
f.write("方法Stab+NN全中的次数为：%d\n" % stab_nn_count)
# 计数次数
stab_nn_count_df = pd.DataFrame(pd.value_counts(stab_nn_names),columns=['count'])
stab_nn_count_df = stab_nn_count_df.reset_index().rename(columns={'index':'id'})
stab_nn_count_df.to_excel(excel_writer=writer, sheet_name='Stab+NN_count', index=False)
stab_nn_dict = featrue_count(list(zip(stab_nn_names, stab_nn_weight )))
# 计数权重
stab_nn_weight_df = pd.DataFrame.from_dict(stab_nn_dict, orient='index', columns=['weight'])
stab_nn_weight_df = stab_nn_weight_df.reset_index().rename(columns={'index':'id'})
stab_nn_weight_df = stab_nn_weight_df.sort_values(by='weight', ascending=False)
stab_nn_weight_df.to_excel(excel_writer=writer, sheet_name='Stab+NN_weight', index=False)

print("svm:",stab_svm_count_df)
print("nn:",stab_nn_count_df)

f.close()
# total_names = lasso_svm_feature_names + lasso_nn_feature_names + stab_svm_feature_names + stab_nn_feature_names
# total_weights = np.concatenate(
#     (lasso_svm_feature_weights, lasso_nn_feature_weights, stab_svm_feature_weights, stab_nn_feature_weights), axis=0)
# selected_result = list(zip(total_names, total_weights))
# print(selected_result)
# dict = featrue_count(selected_result)
# print(dict)
# total_df = pd.DataFrame([lasso_svm_df, lasso_nn_df, stab_svm_df])
# total_df.to_excel(excel_writer=writer, sheet_name='all', index=False)
writer.save()
writer.close()