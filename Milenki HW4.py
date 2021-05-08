import csv
import numpy as np
import math


def main():
    NaiveBayesClassifer()  # Q2
    logisticRegressionClassifier()  # Q3


def NaiveBayesClassifer():
    print("Question 2 Start -------------------")
    # reading data
    name = "spambase.data"
    rawdata = readCSVfile(name)

    # splitting and standardizing
    xTrain, yTrain, xTest, yTest = preProcessData(rawdata)

    # breaking into spam and nonspam groups
    # spam = 1, nonspam = 0
    trainSpamX = []
    trainNonSpamX = []
    for i in range(len(xTrain)):
        if yTrain[i] == 1:
            trainSpamX.append(xTrain[i])
        elif yTrain[i] == 0:
            trainNonSpamX.append(xTrain[i])

    probOfClassSpamTrain = len(trainSpamX) / len(xTrain)  # = P( y = s)
    probOfClassNonSpamTrain = len(trainNonSpamX) / len(xTrain)  # = P (y = ns)

    trainSpamX = np.asarray(trainSpamX)
    trainNonSpamX = np.asarray(trainNonSpamX)

    # calc the means and std for each class
    trainSpamMean = np.mean(trainSpamX, axis=0)  # mean(s,features)
    trainSpamStd = np.std(trainSpamX, axis=0, ddof=1)  # std(s, features)
    trainNonSpamMean = np.mean(trainNonSpamX, axis=0)  # mean(ns, features)
    trainNonSpamStd = np.std(trainNonSpamX, axis=0, ddof=1)  # std(ns, features )

    predictedTestClass = np.zeros(len(xTest))

    # running test
    for i in range(len(xTest)):
        probSpam = np.log(probOfClassSpamTrain)
        probNonSpam = np.log(probOfClassNonSpamTrain)

        # print(i)

        for j in range(len(xTest[0])):
            # print(j)

            logGaussianSpam = np.log(getGuassianProb(xTest[i][j], trainSpamMean[j], trainSpamStd[j]))
            logGaussianNonSpam = np.log(getGuassianProb(xTest[i][j], trainNonSpamMean[j], trainNonSpamStd[j]))

            np.seterr(divide='ignore')  # ignores the popup message for log(0) error

            probSpam = probSpam + logGaussianSpam
            probNonSpam = probNonSpam + logGaussianNonSpam

            # print(xTest[i][j],trainSpamMean[j],trainSpamStd[j])
            # print("Guass Prob Spam:", getGuassianProb(xTest[i][j], trainSpamMean[j], trainSpamStd[j]))
            # print(xTest[i][j], trainNonSpamMean[j], trainNonSpamStd[j])
            # print("Guass Prob NonSpam:", getGuassianProb(xTest[i][j], trainNonSpamMean[j], trainNonSpamStd[j]))

        if probSpam > probNonSpam:
            predictedTestClass[i] = 1
        else:
            predictedTestClass[i] = 0

    # getting accuracy statistics
    # [precision, recall, fmeasure, accuracy]
    sumStats = getSummaryStatistics(predictedTestClass, yTest)

    print("Precision: ", sumStats[0])
    print("Recall: ", sumStats[1])
    print("F measure: ", sumStats[2])
    print("Accuracy: ", sumStats[3])

    print("Question 2 End -------------------")


def logisticRegressionClassifier():
    print("Question 3 Start -------------------")
    print("This takes a few seconds")
    # reading data
    name = "spambase.data"
    # name = "KidCreative.csv"
    rawdata = readCSVfile(name)

    # splitting and standardizing
    xTrain, yTrain, xTest, yTest = preProcessData(rawdata)

    dummyVar = []
    for j in range(len(xTrain)):
        dummyVar.append([1])

    xTrain = np.concatenate((dummyVar, xTrain), axis=1)

    dummyVar = []
    for j in range(len(xTest)):
        dummyVar.append([1])

    xTest = np.concatenate((dummyVar, xTest), axis=1)

    trainedW = gradientAscent(xTrain, yTrain)

    # print(trainedW)
    # running model on testing data
    spamThreshold = .5

    predictedY = np.zeros((len(yTest)))

    for i in range(len(predictedY)):
        g = gFuncSingle(xTest[i], trainedW)
        if g >= spamThreshold:
            predictedY[i] = 1
        else:
            predictedY[i] = 0

    # getting accuracy statistics
    # [precision, recall, fmeasure, accuracy]
    sumStats = getSummaryStatistics(predictedY, yTest)

    print("Precision: ", sumStats[0])
    print("Recall: ", sumStats[1])
    print("F measure: ", sumStats[2])
    print("Accuracy: ", sumStats[3])

    print("Question 3 End -------------------")


def gradientAscent(xTrain, yTrain):
    learnRate = .003
    epsilon = 2 ** -5
    epoch = 0

    prevJ = 100
    # initalize
    Y = yTrain
    X = xTrain
    N = len(X[0])
    w = []

    np.random.seed(0)
    for i in range(N):
        w.append([np.random.uniform(-1, 1)])

    w = np.asarray(w)

    J = calcJ(Y, X, w)

    while abs(J - prevJ) > epsilon:
        epoch += 1
        # print("Epoch: ", epoch )
        prevJ = J
        w = w + (learnRate * calcdJdw(Y, X, w))

        J = calcJ(Y, X, w)

        # print("Change In J: ",abs(J - prevJ))

    return w


def calcJ(Y, X, w):
    J = 0
    for i in range(len(X)):
        g = gFuncSingle(X[i], w)
        if g >= 1:
            g = .999999999
        if g <= 0:
            g = .000000001

        firstTerm = Y[i] * np.log(g)
        secondTerm = (1 - Y[i]) * np.log(1 - g)
        total = firstTerm + secondTerm
        J += total

    return J


def calcdJdw(Y, X, w):
    Yterm = []
    for i in range(len(Y)):
        g = gFuncSingle(X[i], w)
        Yg = Y[i] - g

        Yterm.append([Yg])

    Yterm = np.asarray(Yterm)
    Xt = X.transpose()

    result = np.matmul(Xt, Yterm)
    return result


def gFuncSingle(x, w):
    exponent = -1 * np.matmul(x, w)[0]

    return 1 / (1 + np.exp(exponent))


def getGuassianProb(x, mean, std):
    # return stats.norm.pdf(x,mean,std)
    return (1 / (std * math.sqrt(2 * math.pi))) * math.e ** -((x - mean) ** 2 / (2 * std ** 2))


def preProcessData(data):
    # splits data randomly into training (2/3), testing(1/3)as
    # standardizes the data
    # returns xTrain,yTrain, trainMean, trainStds ,xTest,yTest, testMean, testStds
    np.random.seed(0)
    np.random.shuffle(data)

    data = np.asarray(data)

    X = data[:, :-1]
    Y = data[:, -1]

    # break into train and testing sets
    observations = len(X)
    trainNum = int(observations * (2 / 3))
    testNum = observations - trainNum

    train = X[0:trainNum, :]
    trainAns = Y[0:trainNum]
    trainMeans = np.mean(train, axis=0)
    trainStds = np.std(train, axis=0, ddof=1)

    for i in range(len(train)):
        for j in range(len(train[0])):
            train[i][j] = (train[i][j] - trainMeans[j]) / trainStds[j]

    test = X[trainNum:observations, :]
    testAns = Y[trainNum:observations]
    # testMeans = np.mean(test, axis=0)
    # testStds = np.std(test, axis=0, ddof=1)

    for i in range(len(test)):
        for j in range(len(test[0])):
            # test[i][j] = test[i][j] - testMeans[j] / testStds[j]
            test[i][j] = (test[i][j] - trainMeans[j]) / trainStds[j]

    # return train, trainAns, trainMeans,trainStds, test, testAns, testMeans, testStds
    return np.asarray(train), np.asarray(trainAns), np.asarray(test), np.asarray(testAns)


def getSummaryStatistics(predictedY, actualY):
    confMatrix = getConfusionMatrix(predictedY, actualY)

    precision = calcPrecision(confMatrix)
    recall = calcRecall(confMatrix)
    accuracy = calcAccuracy(confMatrix)
    fmeasure = calcFmeasure(precision, recall)

    return [precision, recall, fmeasure, accuracy]


def getConfusionMatrix(predictedY, trueY):
    # returns [tp,fp,fn,tn]
    N = len(predictedY)
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(N):
        if predictedY[i] == 1 and trueY[i] == 1:
            tp += 1
        elif predictedY[i] == 1 and trueY[i] == 0:
            fp += 1
        elif predictedY[i] == 0 and trueY[i] == 0:
            tn += 1
        elif predictedY[i] == 0 and trueY[i] == 1:
            fn += 1

    return [tp, fp, fn, tn]


def calcPrecision(confusionMatrix):
    # [tp,fp,fn,tn]
    tp = confusionMatrix[0]
    fp = confusionMatrix[1]
    return tp / (tp + fp)


def calcRecall(confusionMatrix):
    # [tp,fp,fn,tn]
    tp = confusionMatrix[0]
    fn = confusionMatrix[2]
    return tp / (tp + fn)


def calcFmeasure(precision, recall):
    return (2 * precision * recall) / (precision + recall)


def calcAccuracy(confusionMatrix):
    # [tp,fp,fn,tn]
    tp = confusionMatrix[0]
    fp = confusionMatrix[1]
    fn = confusionMatrix[2]
    tn = confusionMatrix[3]

    return (tp + tn) / (tp + fp + fn + tn)


def readCSVfile(name):
    with open(name, newline='', encoding="utf-8-sig") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        data = np.asarray(list(csvreader)).astype(np.float64)
    return data


if __name__ == '__main__':
    main()
