import numpy as np

def loadDataSet(fileName):
    fr = np.genfromtxt(fileName, delimiter="\t", dtype=float)
    dataMat=fr[:,0:-1]
    labelMat=fr[:,-1]

    return dataMat,labelMat



def selectJrand(i,m):

    j=i            # 我们要选出所有不等于βi的βj

    while (j==i):

        j = int(np.random.uniform(0,m))

    return j



def clipAlpha(aj,H,L):

    if aj > H: 

        aj = H

    if L > aj:

        aj = L

    return aj



def smoSimple(dataMatIn, classLabels, C, toler, maxIter):     # toler容忍程度，指到边界的距离

    dataMatrix = np.mat(dataMatIn)

    labelMat = np.mat(classLabels).transpose()

    b = 0

    m,n = np.shape(dataMatrix)

    alphas = np.mat(np.zeros((m,1)))

    iter = 0

    while (iter < maxIter):

        alphaPairsChanged = 0

        for i in range(m):    # 选出最不符合条件的βi

            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b

            Ei = fXi - float(labelMat[i])      # 检查是否违反KKT条件

            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):

                j = selectJrand(i,m)

                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b

                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy()

                if (labelMat[i] != labelMat[j]):

                    L = max(0, alphas[j] - alphas[i])

                    H = min(C, C + alphas[j] - alphas[i])

                else:

                    L = max(0, alphas[j] + alphas[i] - C)

                    H = min(C, alphas[j] + alphas[i])

                if L==H:

                    print("L==H")
                    continue

                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T

                if eta >= 0:

                    print("eta>=0")
                    continue

                alphas[j] -= labelMat[j]*(Ei - Ej)/eta

                alphas[j] = clipAlpha(alphas[j],H,L)

                if (abs(alphas[j] - alphaJold) < 0.00001):

                    print("j not moving enough")
                    continue

                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])   # 用与j相同的数量更新i，更新方向相反

                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T

                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T

                if (0 < alphas[i]) and (C > alphas[i]):

                    b = b1

                elif (0 < alphas[j]) and (C > alphas[j]):

                    b = b2

                else:

                    b = (b1 + b2)/2.0

                alphaPairsChanged += 1

                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))

        if (alphaPairsChanged == 0):

            iter += 1

        else:

            iter = 0

        print("iteration number: %d" % iter)

    return b,alphas

if __name__ == '__main__':
    dataMat,labelMat = loadDataSet('../datas/testSet.txt')
    b,alphas = smoSimple(dataMat, labelMat, 0.06, 0.01, 100)
    print('b:',b)
    print('alphas',alphas[alphas>0])