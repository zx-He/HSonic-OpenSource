import numpy as np
from sklearn.svm import SVC
import getDataset
from sklearn.metrics import confusion_matrix
import random


def compute_Metric(ture_label, predict_label):
    conf_matrix = confusion_matrix(ture_label, predict_label)
    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    bac = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    return far, frr, bac

def getTrainAndTest(UserDataset, UserID, trainSampleNum):

    seed = 10000
    train_indices = []
    test_indices = []
    indices = [j for j, label in enumerate(UserDataset.labels) if label == UserID]
    random.seed(seed)
    train_indices.extend(random.sample(indices, trainSampleNum))
    test_indices.extend([index for index in indices if index not in train_indices])
    train_User = getDataset.GetSubset(UserDataset, train_indices)
    test_User = getDataset.GetSubset(UserDataset, test_indices)

    return train_User, test_User


def reTrainDataset(legalDataset, illegalDataset):
    new_samples = []
    new_labels = []

    for sample, label in zip(legalDataset.samples, legalDataset.labels):
        for i in range(37):
            new_samples.append(sample.tolist())
            new_labels.append(1)  # positive sample

    for sample, label in zip(illegalDataset.samples, illegalDataset.labels):
        new_samples.append(sample.tolist())
        new_labels.append(0)  # negative sample

    return new_samples, new_labels

def reTestDataset(legalDataset, illegalDataset):
    new_samples = []
    new_labels = []

    for sample, label in zip(legalDataset.samples, legalDataset.labels):
        new_samples.append(sample.tolist())
        new_labels.append(1)  # positive sample

    for sample, label in zip(illegalDataset.samples, illegalDataset.labels):
        new_samples.append(sample.tolist())
        new_labels.append(0)  # negative sample

    return new_samples, new_labels


if __name__ == '__main__':


    defaultList = []
    for i in range(1, 31):
        defaultList.append(i)
    userList = []
    for i in range(31, 61):
        userList.append(i)

    datasetpath = "/reconstructedFeature/"
    defaultUser = getDataset.getDatasetwithoutDomain(root_dir=datasetpath, UserList=defaultList, dataTypeList=["chair_10", "head_10", "body_10", "walk_10"])


    far_total = 0
    frr_total = 0
    bac_total = 0

    trainSampleNum = 320
    for User in userList:
        legalUser = getDataset.getDatasetwithoutDomain(root_dir=datasetpath, UserList=[User],
                                                dataTypeList=["chair_10", "head_10", "body_10", "walk_10"])
        trainLegal, testLegal = getTrainAndTest(legalUser, User, trainSampleNum)
        train_samples, train_labels = reTrainDataset(trainLegal, defaultUser)
        train_samples = np.array(train_samples).reshape(len(train_samples),-1)
        train_labels = np.array(train_labels)


        IllegalUser = getDataset.getIllegalDataset(root_dir=datasetpath, UserList=userList, legalUser=User,
                                                   dataTypeList=["chair_10", "head_10", "body_10", "walk_10"])
        test_samples, test_labels = reTestDataset(testLegal, IllegalUser)

        test_samples = np.array(test_samples).reshape(len(test_samples),-1)
        test_labels = np.array(test_labels)


        C = 100
        gamma = 0.01
        svm_model = SVC(C=C, gamma=gamma, kernel='rbf')
        svm_model.fit(train_samples, train_labels)

        predictions = svm_model.predict(test_samples)

        far_test, frr_test, bac_test = compute_Metric(ture_label=test_labels, predict_label=predictions)
        far_total = far_total + far_test
        frr_total = frr_total + frr_test
        bac_total = bac_total + bac_test

        print("User, C, gamma, far_test, frr_test, bac_test:", User, C, gamma, far_test, frr_test, bac_test)

    print("far_total, frr_total, bac_total:",
          far_total/len(userList), frr_total/len(userList), bac_total/len(userList))
