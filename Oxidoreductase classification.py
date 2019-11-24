from __future__ import division
from operator import itemgetter
from skimage import feature as ft
from skimage import io
from numpy import *
from sklearn.metrics import confusion_matrix
import cv2

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
import cv2
from sklearn import manifold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file
    returnMat = zeros((numberOfLines - 10, 20))  # prepare matrix to return
    mulMat = zeros((20, 20))
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    n = 0
    index = 0
    for line in fr.readlines():
        if n < 4:
            n = n + 1
        elif n < numberOfLines - 6:
            line = line.strip()
            listFromLine = line.split()
            returnMat[index, :] = listFromLine[2:22]
            classLabelVector.append(int(listFromLine[0]))
            index += 1
            n += 1
    print(returnMat)
    returntMat = returnMat.T
    mulMat = dot(returntMat, returnMat)

    recos=[]
    for i in range(20):
        for j in range(i+1,20):
            re=cos_dist(mulMat[:,i],mulMat[:,j])
            recos.append(re)
    print(shape(recos))
    print(recos)
    recosmat=mat(recos)
    print(shape(recosmat))


    ml, mw = shape(returnMat)
    celll = int(ml / 10)  # 每个cell中的行数
    cellev = 10  # cell数=20
    # print(celll)
    cell_size = 8
    bin_size = 9
    angle_unit = 360 / bin_size
    angmat = zeros((cellev, 9))

    for i in range(cellev):
        cell01 = zeros((celll, 20))
        cell02 = zeros(((ml - (cellev - 1) * celll), 20))
        num = 0
        if i < (cellev - 1):
            for j in range(celll):
                jm = i * celll + j
                cell01[j, :] = returnMat[jm, :]
            gradient_values_x = cv2.Sobel(cell01, cv2.CV_64F, 1, 0, ksize=5)
            gradient_values_y = cv2.Sobel(cell01, cv2.CV_64F, 0, 1, ksize=5)
            mag, angle = cv2.cartToPolar(gradient_values_x, gradient_values_y, angleInDegrees=True)
            # mag = abs(mag)
            cellvector = cell_gradient(mag, angle, bin_size, angle_unit)
            angmat[i, :] = cellvector
            # print("00000")
            # print(angmat.shape)
            # print(angmat)
            # num+=1
        else:
            for j in range((ml - (cellev - 1) * celll)):
                jm = (cellev - 1) * celll + j
                cell02[j, :] = returnMat[jm, :]
            gradient_values_x = cv2.Sobel(cell02, cv2.CV_64F, 1, 0, ksize=5)
            gradient_values_y = cv2.Sobel(cell02, cv2.CV_64F, 0, 1, ksize=5)
            mag, angle = cv2.cartToPolar(gradient_values_x, gradient_values_y, angleInDegrees=True)
            # mag = abs(mag)
            cellvector = cell_gradient(mag, angle, bin_size, angle_unit)
            angmat[9, :] = cellvector
    # print("++++")
    # print(angmat.shape)
    # print(angmat)
    # returntMat=returnMat.T
    # mulMat=dot(returntMat,returnMat)
    # print(mulMat)
    n = shape(mulMat)
    # print(n)
    # angmat = StandardScaler().fit_transform(angmat)
    angmat = angmat.reshape(1, 90)
    angmat = array(angmat)
    # mulMat = StandardScaler().fit_transform(mulMat)
    mulMat = mulMat.reshape(1, 400)
    mulMat = array(mulMat)

    # print(angmat.shape)
    # print(mulMat)
    i = shape(mulMat)
    # print(i)
    # print(mulMat)
    # print(angmat)


    #ftrainmat = np.hstack((mulMat, angmat))
    #ftrainmat = np.hstack((mulMat, recosmat))
    #ftrainmat=np.hstack((ftrainmat,recosmat))
    #print(ftrainmat.shape)
    #print("cosssssssssssssssssssssssssssss")
    #print("cosssssssssssssssssssssssssssss")
    #print("cosssssssssssssssssssssssssssss")
    #print("cosssssssssssssssssssssssssssss")
    #return ftrainmat,angmat,recosmat
    #return ftrainmat
    return mulMat

def percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)  # 升序
    sortArray = sortArray[-1::-1]  # 逆转，即降序
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            print(num)
            return num


def pcatrain(dataMat, percentage=0.99):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    topNfeat = percentage2n(eigVals, percentage)
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    print(lowDDataMat)
    # print(eigVals)
    # print(eigVects)
    # print(redEigVects)
    c = lowDDataMat.shape[1]
    print(c)
    a = shape(lowDDataMat)
    print(a)
    return lowDDataMat, c


def pcatest(dataMat, topNfeat):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet


from sklearn.feature_selection import RFECV
from sklearn import preprocessing
import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from os import listdir
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import chi2


def eachFile(filepath):
    pathDir = listdir(filepath)  # 获取当前路径下的文件名，返回List
    m = len(pathDir)
    trainMat = zeros((m,400))
    #Mat = zeros((m, 90))
    #CMat = zeros((m, 190))
    laber = []
    for i in range(m):
        fileNameStr = pathDir[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('-')[0])
        laber.append(classNumStr)
    print(laber)
    n = 1
    i = 0
    for s in pathDir:
        newDir = os.path.join(filepath, s)  # 将文件命加入到当前文件路径后面
        if os.path.isfile(newDir):  # 如果是文件
            if os.path.splitext(newDir)[1] == ".pssm":  # 判断是否是pssm
                resultvec = file2matrix(newDir)  # 读文件
                trainMat[i, :] = resultvec
                #AMat[i, :] =angmat
                #CMat[i, :] = cosmat
                i += 1
            else:
                pass
    savetxt('pssm.txt', trainMat, delimiter='\t')
    # print("%d"%i)
    # print(laber)
    # print("________________________________________________________________________")
    # print(trainMat)
    m = shape(trainMat)
    # print(m)
    n += 1
    # trainMat=autoNorm(trainMat)
    # trainMat=pcatrain(trainMat)
    # trainMat = pcatrain(trainMat)
    # cross(trainMat,laber)
    # trainMat = SelectKBest(k=300).fit_transform(trainMat, laber)
    # print(shape(trainMat))
    height, width = np.shape(trainMat)
    # gradient_values_x = cv2.Sobel(trainMat, cv2.CV_64F, 1, 0, ksize=5)
    # gradient_values_y = cv2.Sobel(trainMat, cv2.CV_64F, 0, 1, ksize=5)  # tidu
    # img表示源图像，即进行边缘检测的图像
    # cv2.CV_64F表示64位浮点数即64float。
    # 这里不使用numpy.float64，因为可能会发生溢出现象。用cv的数据则会自动
    # 第三和第四个参数分别是对X和Y方向的导数（即dx,dy），这里1表示对X求偏导，0表示不对Y求导。其中，X还可以求2次导。
    # 注意：对X求导就是检测X方向上是否有边缘。
    # 第五个参数ksize是指核的大小。
    # gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    # gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    # print(gradient_magnitude.shape, gradient_angle.shape)
    #mag = abs(trainMat)
    #model1 = SelectKBest(k=400)  # 选择k个最佳特征
    # trainMat=model1.fit_transform(trainMat,laber )
    # clf = ExtraTreesClassifier()
    # clf = clf.fit(trainMat, laber)
    # model = SelectFromModel(clf, prefit=True)
    # trainMat = model.transform(trainMat)






    print(trainMat.shape)
    trainMat = StandardScaler().fit_transform(trainMat)

    #rfe1=RandomForestRegressor();
    #rfe2 = RandomForestRegressor();

    #lr = svm.SVC(kernel='linear', C=0.9)
    # rank all features, i.e continue the elimination until the last one
    #rfe = RFE(estimator=lr,n_features_to_select=116)
    #rfecv1 = RFECV(estimator=lr,  cv=5)
    #rfecv2 = RFECV(estimator=lr, cv=5)
    #rfe.fit(trainMat, laber)
    #print(rfe)
    #rfecv1.fit(trainMat, laber)
    #rfecv1.fit(AMat, laber)
    #mask1 = rfecv1.get_support()
    #print(mask1)
    #print("Optimal number of features : %d" % rfecv1.n_features_)
    #rfecv2.fit(CMat, laber)
    #rfe.fit(trainMat, laber)

    #trainMat = rfe.transform(trainMat)
    #print(trainMat.shape)
    #mask = rfe.get_support()
    #print(mask)
    #svm2()
    #mask2 = rfecv2.get_support()
    #print(mask2)

    #plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    #print(rfe.n_features_)
    #print(rfe.n_features_)
    #print(rfe.ranking_)

    #print("Optimal number of features : %d" % rfecv2.n_features_)


    #tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #trainMat = tsne.fit_transform(trainMat)
    #plt.plot()
    #plt.scatter(x=trainMat[:, 0], y=trainMat[:, 1])  # 画散点图
    #plt.show()
    #kmeans(trainMat)
    # meanshift(trainMat)
    # plot_embedding(trainMat)
    # plt.show()
    # min_max_scaler = preprocessing.MinMaxScaler()
    # trainMat = min_max_scaler.fit_transform(trainMat)
    #lr = svm.SVC(kernel='linear',C=0.9)
    # rank all features, i.e continue the elimination until the last one
    #rfe = RFE(estimator=lr)
    #rfecv = RFECV(estimator=lr,  cv=5)
    #rfe.fit(trainMat, laber)
    #trainMat=rfe.transform(trainMat)
    #print(trainMat.shape)
    # print(rfe.n_features_)
    # print(rfe.ranking_)
    # rfe.transform(trainMat,laber)
    # rfe.transform(trainMat)
    print(trainMat)
    x_train, x_test, y_train, y_test = train_test_split(trainMat, laber, random_state=1, train_size=0.8)
    #adaboost(x_train, y_train, x_test, y_test)
    #KNN(x_train, x_test, y_train, y_test)
    #print(x_train)
    #print(y_train)
    #tsnelaber = np.mat(y_train)
    #tsnelaber = tsnelaber / 100
    #tsnelaber = tsnelaber.T
    #x_train = np.hstack((x_train, tsnelaber))
    #print(x_train)
    #print(x_train.shape)
    #x_train = array(x_train)
    #print("_______________________________________________________")
    #trainclu0, trainclu1, trainclu2, trainclu3, clu0laber, clu1laber, clu2laber, clu3laber,center=kmeans(x_train)
    #ypre=[]
    #sunn=0
    #i=0
    #clfs1,clfs2,clfs3,clfs4=DONGFen(trainclu0, trainclu1, trainclu2, trainclu3, clu0laber, clu1laber, clu2laber, clu3laber, center)
    #for eachtest in x_test:
        #presult=predong(clfs1,clfs2,clfs3,clfs4,center,eachtest)
        #print(presult)
        #if(presult==y_test[i]):
            #sunn=sunn+1
        #i+=1
        #ypre.append(int(presult[0]))
    #conf_mat = confusion_matrix(y_test, ypre)
    #print(conf_mat)
    #print(sunn)
    #print(y_test)
    # W=fwknn(x_train, x_test, y_train, y_test)
    #all = [314, 215, 194, 130, 112, 305, 64, 59, 254, 94, 154, 94, 257, 155, 84, 154]
    #TP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #FP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #agomat=[]
    #nowmat=[]
    #for i in range(2639):
        #ago, now = leaveoneout2(trainMat, laber, i)
        #agomat.append(ago)
        #nowmat.append(now)
        #d = int(ago)
        #b = int(now)
        #if (ago == now):
         #   TP[d - 1] += 1
        #for a in range(1, 17):
        #    if ((ago != now) & (now == a)):
        #        FP[a - 1] += 1
        #print(TP)
        #print(FP)
        #print(i)
        #print("________________________________________________")
    #conf_mat = confusion_matrix(agomat, nowmat)
    #print(conf_mat)

    # x_train,q=pcatrain(x_train)
    # print(q)
    # x_train=x_train.astype(complex)
    # print(x_train)
    # x_test=pcatest(x_test,q)
    # x_test=x_test.astype(complex)
    # print(y_train)
    # a=shape(x_train)
    # print(a)
    # b=shape(x_test)
    # print(b)
    # x_train=pca(x_train)
    # x_test=pca(x_test)
    # print(pcamat)
    # c=shape(pcamat)
    # print(c)
    # i=2
    # wa=[]
    # while i<16:
    # b=adaboost(x_train,y_train,x_test,y_test,i)
    # i+=2
    # wa.append(b)
    # print(wa)
    # w1,w2,w3=weight(x_train,y_train,x_test,y_test)
    # w11, w22, w33 = confuse(x_train, y_train, x_test, y_test)
    # weightvote(x_train, y_train, x_test, y_test,w1,w2,w3,w11,w22,w33)


    # vote(x_train, y_train, x_test, y_test,trainMat, laber,W)
    # baggingknn(x_train, y_train, x_test, y_test,trainMat,laber)
    #adaboost(x_train, y_train, x_test, y_test)

    ago,now=svm2(x_train,x_test,y_train,y_test)
    conf_mat = confusion_matrix(ago, now)
    print(conf_mat)
    #KNN(x_train, x_test, y_train, y_test)
    # classify0(x_test,x_train,y_train,1)


def svm2(x_train, x_test, y_train, y_test):
    #clf = svm.SVC(C=0.9, kernel='linear', decision_function_shape='ovr',probability=True)
    clf = svm.SVC(C=200, kernel='rbf', gamma=0.01, decision_function_shape='ovr', probability=True)  #hog最佳参数   0.9602
    #clf = svm.SVC(C=150, kernel='rbf', gamma=0.001, decision_function_shape='ovr', probability=True)  #cos最佳参数 0.9621
    #clf = svm.SVC(C=100, kernel='rbf', gamma=0.01, decision_function_shape='ovr', probability=True)  #pssm最佳   0.9621
    #clf = svm.SVC(C=c, kernel='rbf', gamma=g, decision_function_shape='ovr', probability=True)
    clf.fit(x_train, y_train)
    #clf.score(x_train, y_train)

    # y_hat = clf.predict(x_train)
    # print(y_hat)
    ac=clf.score(x_test, y_test)
    print(ac)
    y_hat = clf.predict(x_test)
    # print(y_hat)
   # pro = clf.predict_proba(x_test)
    print('svm')
    #print(pro)
    #return pro
    return y_test,y_hat


def makecovmat(trainMat):
    # for i in range(3):
    # for j in range(3):
    Y = cov(trainMat, rowvar=0)
    # print(z)
    # X = [trainMat[0], trainMat[1], trainMat[2]]
    # Y = np.cov(X)
    print(Y)
    a = shape(Y)
    print(Y)
    return Y


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def KNN(train_data, test_data, train_target, test_target):
    clf = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=3)
    clf.fit(train_data, train_target)
    print(clf)
    test_res = clf.predict(test_data)
    print(test_target)
    print(test_res)
    # 打印预测准确率
    pro = clf.predict_proba(test_data)
    print('knn')
    print(pro)
    print(accuracy_score(test_res, test_target))
    return pro


from sklearn.cross_validation import cross_val_score


def cross(X, y):
    # clf=KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=2)
    clf = svm.SVC(C=0.9, kernel='linear', decision_function_shape='ovr')
    # svm.SVC(C=0.9, kernel='rbf', gamma=300, decision_function_shape='ovr')
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print(scores)
    print(scores.mean())


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier


def adaboost(trainmat, laber, testmat, testlaber):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),
                             algorithm="SAMME", n_estimators=30,
                             learning_rate=0.9)
    clf.fit(trainmat, laber)
    scores1 = clf.score(trainmat, laber)
    # result=clf.predict(testmat)
    scores = clf.score(testmat, testlaber)
    print(scores)
    # a=scores.mean()
    # pro = clf.predict_proba(testmat.reshape((1,-1)))
    # print(result)
    # print(scores1)
    # print(a)
    print('ada')
    # print(pro)
    # return pro


def leaveoneout2(trainmat, laber, i):
    tmat = zeros((1,400))
    print(shape(laber))
    llaber = laber[:]
    testmat = trainmat[i]
    tmat[0, :] = testmat
    testlaber = llaber[i]
    trainmat2 = delete(trainmat, i, axis=0)
    llaber.pop(i)
    print(tmat)
    a = shape(llaber)
    print(a)



    print("_______________________________________________________")

    #w1, w2, w3 = weight(trainmat2, llaber, tmat, testlaber)
    #result = weightvote(trainmat2, llaber, tmat, testlaber, w1, w2, w3)
    # result=vote(trainmat2,llaber,tmat,testlaber)
    # result=baggingknn(trainmat2,llaber,tmat,testlaber)
    # result=Wknnclass(trainmat2, llaber, tmat, 1,w)
    # result=adaboost(trainmat2,llaber,tmat,testlaber)
    #result=KNN(trainmat2,tmat,llaber,testlaber)
    result=svm2(trainmat2,tmat,llaber,testlaber)
    return testlaber, result


def knnclass(dataSet, labels, testmat, k):
    dataSetSize = dataSet.shape[0]
    difmat = (testmat - dataSet)
    # print(difmat)
    # print(shape(difmat))
    sqDiffMat = difmat ** 2
    # print(sqDiffMat)
    # print(shape(sqDiffMat))
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # print(distances)
    sortedDistIndicies = distances.argsort()
    # print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)
    # print(sortedClassCount[0][0])
    return sortedClassCount[0][0]


def Wknnclass(dataSet, labels, testmat, k, w):
    dataSetSize = dataSet.shape[0]
    difmat = (testmat - dataSet)
    # print(difmat)
    # print(shape(difmat))
    sqDiffMat = difmat ** 2
    sqDiffMat = sqDiffMat * w
    # sqDiffMat=[M * N for M, N in zip(sqDiffMat, w)]
    # print(sqDiffMat)
    # print(shape(sqDiffMat))
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # print(distances)
    sortedDistIndicies = distances.argsort()
    # print(sortedDistIndicies)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)
    # print(sortedClassCount[0][0])
    return sortedClassCount[0][0]


def fwknn(x_train, x_test, y_train, y_test):
    # trainmat=x_train[:]
    # testmat=x_test[:]
    # trainlaber=y_train[:]
    # tlaber=y_test[:]
    result = []
    i = 0
    errorsum = 0
    er2mat = []
    for eschmat in x_test:  # all
        r = knnclass(x_train, y_train, eschmat, 1)
        result.append(r)
        if (y_test[i] != r):
            errorsum += 1
        i += 1
    print(errorsum)
    for n in range(400):  # every feature
        trainmat = x_train[:]
        testmat = x_test[:]
        trainlaber = y_train[:]
        trainmat2 = delete(trainmat, n, axis=1)
        testmat2 = delete(testmat, n, axis=1)
        j = 0
        errorsum2 = 0
        for eschmat in testmat2:  # delate
            y = knnclass(trainmat2, trainlaber, eschmat, 1)
            # result.append(r)
            if (y_test[j] != y):
                errorsum2 += 1
            j += 1
        er2mat.append(errorsum2)
        print('1')
        print(er2mat)
        print(shape(er2mat))
    for o in range(400):
        er2mat[o] = (er2mat[o] / errorsum) ** 2
    print(er2mat)
    # er2mat/errorsum
    usum = np.sum(er2mat)
    w = []
    for m in range(400):
        wi = er2mat[m] / usum
        w.append(wi)
    print(w)
    est = []
    eserrorsum = 0
    t = 0
    for efile in x_test:
        es = Wknnclass(x_train, y_train, efile, 1, w)
        est.append(es)
        if (y_test[t] != es):
            eserrorsum += 1
        t += 1
    print(eserrorsum)
    return w


from sklearn.ensemble import BaggingClassifier


def baggingknn(trainmat, laber, testmat, tlaber, mat, mlaber):
    clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),
                              algorithm="SAMME", n_estimators=30, learning_rate=0.9)
    clf = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=1)
    # clf =svm.SVC(C=500, kernel='rbf', gamma=0.001, decision_function_shape='ovr')
    clfb = BaggingClassifier(base_estimator=clf1
                             , max_samples=1.0, max_features=1.0, n_estimators=20)
    clfb.fit(trainmat, laber)

    # predict = clf.predict(trainmat)
    # result= clfb.predict(testmat)

    # print(clf.score(trainmat,laber))
    # print(clf.score(testmat,tlaber))
    # result2=clfb.score(testmat, tlaber)
    score = clfb.score(testmat, tlaber)
    print(clfb.score(testmat, tlaber))
    # print(result)
    score1c1 = cross_val_score(clf, mat, mlaber, cv=5, scoring='accuracy')
    scorec2 = cross_val_score(clfb, mat, mlaber, cv=5, scoring='accuracy')
    print('knn')
    print(score1c1.mean())
    print('bagging')
    print(scorec2.mean())
    return score
    # print Series(predict).value_counts()
    # print Series(predictb).value_counts()


from sklearn.ensemble import RandomForestClassifier


def vote(trainmat, laber, testmat, llaber, W):
    clf1 = svm.SVC(C=500, kernel='rbf', gamma=0.001, decision_function_shape='ovr')
    clf2 = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=1)
    clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),
                              algorithm="SAMME", n_estimators=30,
                              learning_rate=0.9)
    clf4 = RandomForestClassifier(min_samples_split=2, n_estimators=140, min_samples_leaf=2, max_depth=14,
                                  random_state=60)
    clf5 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=150, min_samples_split=2, min_samples_leaf=20,
                                      max_depth=16, subsample=0.8, random_state=10)
    # W = fwknn(trainmat, testmat, laber, llaber)
    clf4 = Wknnclass(trainmat, laber, testmat, 1, W)

    # eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('tree', clf3),('rand',clf4)],voting='hard')
    # eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('tree', clf3)], voting='hard')
    eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('rand', clf4)], voting='hard')
    # eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('gbdt', clf5)], voting='hard')
    # eclf1 = VotingClassifier(estimators=[('knn', clf1), ('tree', clf3), ('rand', clf4)], voting='hard')
    # eclf1 = VotingClassifier(estimators=[('svm', clf1), ('knn', clf2), ('tree', clf3)], voting='soft',weights=[0.95,0.96,0.956],flatten_transform=True,)
    eclf1.fit(trainmat, laber)
    result = eclf1.predict(testmat)
    # score=eclf1.score(testmat,llaber)
    # score=eclf1.predict_proba(testmat)
    # clf=clf2.fit(trainmat,laber)
    # print("knn")
    # print(clf.score(testmat,llaber))
    # print("vote")
    # print(score)
    # score1c1=cross_val_score(clf,mat,mlaber,cv=10,scoring='accuracy')
    # scorec2=cross_val_score(eclf1,mat,mlaber,cv=10,scoring='accuracy')
    # print('knn')
    # print(score1c1.mean())
    # print('vote')
    # print(scorec2.mean())
    return result


import math


def weight(trainmat, laber, testmat, llaber):
    clf1 = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=3)
    clf1.fit(trainmat, laber)
    p1 = clf1.score(trainmat, laber)
    print(p1)

    clf2 = svm.SVC(C=50, kernel='rbf', gamma=0.01, decision_function_shape='ovr')
    clf2.fit(trainmat, laber)
    p2 = clf2.score(trainmat, laber)  # 精度
    print(p2)

    clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),
                              algorithm="SAMME", n_estimators=30,
                              learning_rate=0.9)
    clf3.fit(trainmat, laber)
    p3 = clf3.score(trainmat, laber)
    print(p3)

    m1 = p1 * (1.01 - p1)
    m2 = p2 * (1.01 - p2)
    m3 = p3 * (1.01 - p3)
    sumlog = m1 + m2 + m3
    logw1 = math.log(m1)
    logw2 = math.log(m2)
    logw3 = math.log(m3)

    w1 = logw1 / sumlog
    w2 = logw2 / sumlog
    w3 = logw3 / sumlog
    print(w1, w2, w3)

    sunW = w1 + w2 + w3
    w11 = w1 / sunW
    w22 = w2 / sunW
    w33 = w3 / sunW
    print(w11, w22, w33)
    return w11, w22, w33


def weightvote(trainmat, laber, testmat, llaber, w1, w2, w3, w11, w22, w33):
    l = 0
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for each in testmat:
        p = []
        p11 = []
        p22 = []
        p33 = []
        p111 = []
        p222 = []
        p333 = []
        pp = []
        testlaber = llaber[l]
        p1 = KNN(trainmat, each, laber, testlaber)
        for i in range(16):
            p11.append(w11[i] * p1[0][i])
        print(p11)
        print(w1)
        for each1 in p11:
            p111.append(each1 * w1)
        # p11=p11*w1
        print(p111)
        p2 = svm2(trainmat, each, laber, testlaber)
        for i in range(16):
            p22.append(float(w22[i] * p2[0][i]))
        print(p22)
        print(w2)
        for each2 in p22:
            p222.append(each2 * w2)
        # p11=p11*w1
        print(p222)
        # p22=p22*w2
        # m2=p2.sum()
        # print(m2)
        # p2=p2*(1/m2)
        # print(p2)
        p3 = adaboost(trainmat, laber, each, testlaber)
        for i in range(16):
            p33.append(w33[i] * p3[0][i])
        for each3 in p33:
            p333.append(each3 * w3)
        # p11=p11*w1
        print(p333)
        # p33=p33*w3
        # m3 = p3.sum()
        # print(m3)
        # p3 = p3 * (1 / m3)
        # print(p3)
        for i in range(16):
            pall = p111[i] + p222[i] + p333[i]
            p.append(pall)
        # p=p111+p222+p333
        print(p)
        print(shape(p))
        print("+++++++++++++++++++++++++")
        for i in range(16):
            a = p[i]
            pp.append(a)
        print(pp)
        vlaber = pp.index(max(pp))
        vlaber = vlaber + 1
        print(vlaber)
        if (testlaber == vlaber):
            result[vlaber - 1] += 1
        l += 1
        print(result)
    return vlaber


from sklearn.metrics import confusion_matrix, accuracy_score


def confuse(trainmat, laber, testmat, tlaber):
    c1mat = zeros((16, 16))
    c2mat = zeros((16, 16))
    c3mat = zeros((16, 16))

    clf1 = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=3)
    clf1.fit(trainmat, laber)
    test_res1 = clf1.predict(trainmat)
    print(test_res1)
    print(tlaber)
    C1 = confusion_matrix(laber, test_res1)
    print(C1)
    print(shape(C1))
    k = []
    for i in range(16):
        sum1 = 0
        for j in range(16):
            sum1 += C1[i][j]
        k.append(sum1)
        a1 = C1[i][:] / sum1
        c1mat[i, :] = a1
    print(c1mat)  # p(s/k)
    print(k)  # k[]lei
    sumk = sum(k)
    print(sumk)
    pk = k / sumk
    print(pk)  # p(k)
    c11mat = zeros((16, 16))
    for i in range(16):
        a11 = c1mat[i][:] * pk[i]
        c11mat[i, :] = a11  # p(k)p(s/k)
    print(c11mat)

    print("___________________")

    clf2 = svm.SVC(C=50, kernel='rbf', gamma=0.01, decision_function_shape='ovr', probability=True)
    clf2.fit(trainmat, laber)
    test_res2 = clf2.predict(trainmat)
    print(test_res2)
    print(tlaber)
    C2 = confusion_matrix(laber, test_res2)
    print(C2)
    print(shape(C2))
    for i in range(16):
        sum2 = 0
        for j in range(16):
            sum2 += C2[i][j]
        a2 = C2[i][:] / sum2
        c2mat[i, :] = a2
    print(c2mat)

    c22mat = zeros((16, 16))
    for i in range(16):
        a22 = c2mat[i][:] * pk[i]
        c22mat[i, :] = a22  # p(k)p(s/k)
    print(c22mat)

    clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),
                              algorithm="SAMME", n_estimators=30,
                              learning_rate=0.9)
    clf3.fit(trainmat, laber)
    test_res3 = clf3.predict(trainmat)
    print(test_res3)
    print(tlaber)
    C3 = confusion_matrix(laber, test_res3)
    print(C3)
    print(shape(C3))
    for i in range(16):
        sum3 = 0
        for j in range(16):
            sum3 += C3[i][j]
        a3 = C1[i][:] / sum3
        c3mat[i, :] = a3
    print(c3mat)
    c33mat = zeros((16, 16))
    for i in range(16):
        a33 = c3mat[i][:] * pk[i]
        c33mat[i, :] = a33  # p(k)p(s/k)
    print(c33mat)

    W1 = []
    W2 = []
    W3 = []
    for n in range(16):
        w1 = c11mat[n][n]
        w2 = c22mat[n][n]
        w3 = c33mat[n][n]
        W1.append(w1)
        W2.append(w2)
        W3.append(w3)
    print(W1)
    print(W2)
    print(W3)
    return W1, W2, W3


from sklearn.cluster import k_means, KMeans


def pic(trainmat, laber,c0,c1,c2,c3):

    m, n = shape(trainmat)
    clu0 = zeros((c0, 3))
    clu1 = zeros((c1, 3))
    clu2 = zeros((c2, 3))
    clu3 = zeros((c3, 3))
    #clu4 = zeros((800, 2))
    #clu5 = zeros((800, 2))
    h = 0
    a = 0
    b = 0
    c = 0
    #d = 0
    #z = 0
    for n in range(m):
        if laber[n] == 0:
            clu0[h, :] = trainmat[n, :]
            h += 1

    print(clu0)
    clu0laber=clu0[:,-1]
    clu0laber = [int(x * 100) for x in clu0laber]
    trainclu0=np.delete(clu0,-1,axis=1)
    print(clu0laber)
    print(len(clu0laber))
    print(trainclu0.shape)



    print(clu0.shape)
    for n in range(m):
        if laber[n] == 1:
            clu1[b, :] = trainmat[n, :]
            b += 1
    print(clu1)
    clu1laber = clu1[:, -1]
    clu1laber = [int(x * 100) for x in clu1laber]
    trainclu1 = np.delete(clu1, -1, axis=1)
    print(clu1laber)
    print(len(clu1laber))
    print(trainclu1.shape)

    print(clu1.shape)
    for n in range(m):
        if laber[n] == 2:
            clu2[c, :] = trainmat[n, :]
            c += 1
    print(clu2)
    print(clu2.shape)
    clu2laber = clu2[:, -1]
    clu2laber = [int(x * 100) for x in clu2laber]
    trainclu2 = np.delete(clu2, -1, axis=1)
    print(clu2laber)
    print(len(clu2laber))
    print(trainclu2.shape)

    for n in range(m):
        if laber[n] == 3:
            clu3[a, :] = trainmat[n, :]
            a += 1
    print(clu3)
    print(clu3.shape)
    clu3laber = clu3[:, -1]
    clu3laber = [int(x * 100) for x in clu3laber]
    trainclu3 = np.delete(clu3, -1, axis=1)
    print(clu3laber)
    print(len(clu3laber))
    print(trainclu3.shape)

    # for n in range(m):
    # if laber[n]==4:
    # clu4[a,:]=trainmat[n,:]
    # d+=1
    # f1 = plt.figure(1)  # 创建显示图形输出的窗口对象
    # for n in range(m):
    # if laber[n] == 5:
    # clu5[a, :] = trainmat[n, :]
    # z += 1
    plt.plot()
    plt.scatter(x=clu0[:, 0].tolist(), y=clu0[:, 1].tolist(), edgecolors='r')  # 画散点图
    plt.scatter(x=clu1[:, 0].tolist(), y=clu1[:, 1].tolist(), edgecolors='g')
    plt.scatter(x=clu2[:, 0].tolist(), y=clu2[:, 1].tolist(), edgecolors='b')
    plt.scatter(x=clu3[:, 0].tolist(), y=clu3[:, 1].tolist(), edgecolors='y')
    # plt.scatter(x=clu4[:, 0], y=clu4[:, 1], edgecolors='gray')
    # plt.scatter(x=clu4[:, 0], y=clu4[:, 1], edgecolors='orange')
    #plt.show()
    #DONGFen(trainclu0,trainclu1,trainclu2,trainclu3,clu0laber,clu1laber,clu2laber,clu3laber)
    return trainclu0,trainclu1,trainclu2,trainclu3,clu0laber,clu1laber,clu2laber,clu3laber

def kmeans(trainmat):
    # clf=k_means(trainmat,n_clusters=3,max_iter=10,return_n_iter=True)
    m=len(trainmat)
    clf = KMeans(n_clusters=4)
    clf.fit(trainmat)
    klaber = clf.labels_
    center = clf.cluster_centers_
    center=np.delete(center,-1,axis=1)
    print(clf)
    print(klaber)
    print(center)
    clusters = clf.labels_.tolist()
    print(clusters)
    c0 = 0
    c1 = 0
    c2 = 0
    c3 = 0
    for i in range(m):
        if clusters[i] == 0:
            c0 += 1
        if clusters[i] == 1:
            c1 += 1
        if clusters[i] == 2:
            c2 += 1
        if clusters[i] == 3:
            c3 += 1
    print(c0)
    print(c1)
    print(c2)
    print(c3)

    print(len(clusters))
    # trainmat=mat(trainmat)
    f1 = plt.figure(1)  # 创建显示图形输出的窗口对象
    plt.plot()
    plt.scatter(x=trainmat[:, 0], y=trainmat[:, 1])  # 画散点图
    print(trainmat[:, 0])
    #plt.show()
    trainclu0, trainclu1, trainclu2, trainclu3, clu0laber, clu1laber, clu2laber, clu3laber=pic(trainmat, klaber,c0,c1,c2,c3)
    return trainclu0, trainclu1, trainclu2, trainclu3, clu0laber, clu1laber, clu2laber, clu3laber,center



from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs


def meanshift(trainmat):
    bandwidth = estimate_bandwidth(trainmat, quantile=0.2, n_samples=50)
    clf = MeanShift(bandwidth=bandwidth)
    clf.fit(trainmat)
    cl = clf.fit_predict(trainmat)
    print(cl)
    plt.title('Mean Shift')
    klaber = clf.labels_
    center = clf.cluster_centers_
    print(clf)
    print(klaber)
    print(center)
    # trainmat=mat(trainmat)
    f1 = plt.figure(1)  # 创建显示图形输出的窗口对象
    plt.plot()
    plt.scatter(x=trainmat[:, 0], y=trainmat[:, 1])  # 画散点图
    plt.show()
    pic(trainmat, klaber)


def cell_gradient(cell_magnitude, cell_angle, bin_size, angle_unit):
    orientation_centers = [0] * bin_size
    # [0,0,0,0,0,0,0,0,0]
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]  # 幅度
            gradient_angle = cell_angle[k][l]  # 角度

            # 开始分配每一个所属的柱状
            min_angle = int(gradient_angle / angle_unit) % 9
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    # print(orientation_centers)
    l = shape(orientation_centers)
    # print(l)
    return orientation_centers


from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
def DONGFen(trainmat1,trainmat2,trainmat3,trainmat4,laber1,laber2,laber3,laber4,center):
    print(center)
    Kclf1 = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=3)
    Kclf2 = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=3)
    Kclf3 = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=3)
    Kclf4 = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=3)
    Vclf1 = svm.SVC(C=210, kernel='rbf', gamma=0.001, decision_function_shape='ovr', probability=True)
    Vclf2 = svm.SVC(C=210, kernel='rbf', gamma=0.001, decision_function_shape='ovr', probability=True)
    Vclf3 = svm.SVC(C=210, kernel='rbf', gamma=0.001, decision_function_shape='ovr', probability=True)
    Vclf4 = svm.SVC(C=210, kernel='rbf', gamma=0.001, decision_function_shape='ovr', probability=True)
    Aclf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),algorithm="SAMME", n_estimators=30,learning_rate=0.9)
    Aclf2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),algorithm="SAMME", n_estimators=30, learning_rate=0.9)
    Aclf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),algorithm="SAMME", n_estimators=30, learning_rate=0.9)
    Aclf4 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6), algorithm="SAMME", n_estimators=30, learning_rate=0.9)
    Lclf1 = svm.SVC(C=0.9, kernel='linear', decision_function_shape='ovr', probability=True)
    Lclf2 = svm.SVC(C=0.9, kernel='linear', decision_function_shape='ovr', probability=True)
    Lclf3 = svm.SVC(C=0.9, kernel='linear', decision_function_shape='ovr', probability=True)
    Lclf4 = svm.SVC(C=0.9, kernel='linear', decision_function_shape='ovr', probability=True)

    CLF1=[]
    CLF2=[]
    CLF3=[]
    CLF4=[]


    Kclf1.fit(trainmat1, laber1)
    Kclf2.fit(trainmat2, laber2)
    Kclf3.fit(trainmat3, laber3)
    Kclf4.fit(trainmat4, laber4)

    Vclf1.fit(trainmat1,laber1)
    Vclf2.fit(trainmat2, laber2)
    Vclf3.fit(trainmat3, laber3)
    Vclf4.fit(trainmat4, laber4)

    Aclf1.fit(trainmat1,laber1)
    Aclf2.fit(trainmat2, laber2)
    Aclf3.fit(trainmat3, laber3)
    Aclf4.fit(trainmat4, laber4)

    #Lclf1.fit(trainmat1, laber1)
    #Lclf2.fit(trainmat2, laber2)
    #Lclf3.fit(trainmat3, laber3)
    #Lclf4.fit(trainmat4, laber4)

    CLF1.append(Kclf1)
    CLF1.append(Vclf1)
    CLF1.append(Aclf1)
    #CLF1.append(Lclf1)

    CLF2.append(Kclf2)
    CLF2.append(Vclf2)
    CLF2.append(Aclf2)
    #CLF2.append(Lclf2)

    CLF3.append(Kclf3)
    CLF3.append(Vclf3)
    CLF3.append(Aclf3)
    #CLF3.append(Lclf3)

    CLF4.append(Kclf4)
    CLF4.append(Vclf4)
    CLF4.append(Aclf4)
    #CLF4.append(Lclf4)

    score1=  []
    score2 = []
    score3 = []
    score4 = []

    for ca1 in CLF1:
        score=ca1.score(trainmat1,laber1)
        score1.append(score)
    print(score1)
    sc1sort1=np.argsort(score1)
    sc1max1index=sc1sort1[-1]
    sc1max2index=sc1sort1[(len(sc1sort1)-2)]
    sc1max3index = sc1sort1[(len(sc1sort1) - 3)]
    best1CLF1=  CLF1[sc1max1index]
    best2CLF1 = CLF1[sc1max2index]
    best3CLF1 = CLF1[sc1max3index]
    print(best1CLF1)
    print(best2CLF1)
    print(best3CLF1)

    for ca2 in CLF2:
        score = ca2.score(trainmat2, laber2)
        score2.append(score)
    print(score2)
    sc1sort2 = np.argsort(score2)
    sc2max1index = sc1sort2[-1]
    sc2max2index = sc1sort2[(len(sc1sort2) - 2)]
    sc2max3index = sc1sort2[(len(sc1sort2) - 3)]
    best1CLF2 = CLF2[sc2max1index]
    best2CLF2 = CLF2[sc2max2index]
    best3CLF2 = CLF2[sc2max3index]

    for ca3 in CLF3:
        score = ca3.score(trainmat3, laber3)
        score3.append(score)
    print(score3)
    sc1sort3 = np.argsort(score3)
    sc3max1index = sc1sort3[-1]
    sc3max2index = sc1sort3[(len(sc1sort3) - 2)]
    sc3max3index = sc1sort3[(len(sc1sort3) - 3)]
    best1CLF3 = CLF3[sc3max1index]
    best2CLF3 = CLF3[sc3max2index]
    best3CLF3 = CLF3[sc3max3index]

    for ca4 in CLF4:
        score = ca4.score(trainmat4, laber4)
        score4.append(score)
    print(score4)
    sc1sort4 = np.argsort(score4)
    sc4max1index = sc1sort4[-1]
    sc4max2index = sc1sort4[(len(sc1sort4) - 2)]
    sc4max3index = sc1sort4[(len(sc1sort4) - 3)]
    best1CLF4 = CLF4[sc4max1index]
    best2CLF4 = CLF4[sc4max2index]
    best3CLF4 = CLF4[sc4max3index]

    clfL = LogisticRegression(C=100)

    clfs1 = StackingClassifier(classifiers=[best1CLF1,best2CLF1], use_probas=True, average_probas=True,meta_classifier=clfL)
    clfs2 = StackingClassifier(classifiers=[best1CLF2, best2CLF2], use_probas=True, average_probas=True, meta_classifier=clfL)
    clfs3 = StackingClassifier(classifiers=[best1CLF3, best2CLF3], use_probas=True, average_probas=True,meta_classifier=clfL)
    clfs4 = StackingClassifier(classifiers=[best1CLF4, best2CLF4], use_probas=True, average_probas=True, meta_classifier=clfL)

    clfs1.fit(trainmat1,laber1)
    clfs2.fit(trainmat2, laber2)
    clfs3.fit(trainmat3, laber3)
    clfs4.fit(trainmat4, laber4)
    return clfs1,clfs2,clfs3,clfs4

def predong(clfs1,clfs2,clfs3,clfs4,center,testmat):
    distance=[]
    lc=len(center)
    for i in range(lc):
        dis=np.linalg.norm(center[i,:] - testmat)
        distance.append(dis)
    close=distance.index(min(distance))
    if(close==0):
        #result = best1CLF1.predict(testmat.reshape((1, -1)))
        result=clfs1.predict(testmat.reshape((1,-1)))
    elif(close==1):
        #result = best1CLF2.predict(testmat.reshape((1, -1)))
        result=clfs2.predict(testmat.reshape((1,-1)))
    elif(close==2):
        #result = best1CLF3.predict(testmat.reshape((1, -1)))
        result=clfs3.predict(testmat.reshape((1,-1)))
    else:
        #result = best1CLF4.predict(testmat.reshape((1, -1)))
        result=clfs4.predict(testmat.reshape((1,-1)))
    return result

import math
def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a, b):
        part_up += a1 * b1
        a_sq += a1 ** 2
        b_sq += b1 ** 2
    part_down = math.sqrt(a_sq * b_sq)
    return part_up / part_down

eachFile('z')
# file2matrix('11-1.pssm')




