
import getDataset
from torch.utils.data import Dataset
import torch
from BANN import BANNModel
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler
import random

def trainer(my_net, data_source, data_domain):
    data, class_label = data_source

    my_net.zero_grad()
    batch_size = len(class_label)

    domain_label = torch.zeros(batch_size, dtype=torch.long)
    for i in range(len(domain_label)):
        domain_label[i] = int(data_domain)

    if cuda:
        data = data.cuda()
        class_label = class_label.cuda()
        domain_label = domain_label.cuda()

    class_output, domain_output = my_net(input_data=data, lamuda=lamuda)

    loss_s_class = loss_class(class_output, class_label)
    loss_s_domain = loss_domain(domain_output, domain_label)

    return loss_s_class, loss_s_domain



class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return sample, label

    def __len__(self):
        return len(self.samples)

def reDataset(dataset, userID, UserNum):
    new_samples = []
    new_labels = []
    for sample, label in zip(dataset.samples, dataset.labels):
        if label == userID:
            for _ in range(UserNum-1):
                new_samples.append(sample)
                new_labels.append(0) #positive sample
        else:
            new_samples.append(sample)
            new_labels.append(1) #negative sample
    new_dataset = CustomDataset(new_samples, new_labels)

    return new_dataset


def get_Train_Test_Dataset(dataset, UserList, trainNum = 80):

    subset1_Num = trainNum
    subset2_Num = 100-trainNum

    subset1_index = []
    subset2_index = []

    for label in UserList:
        class_samples = [i for i, l in enumerate(dataset.labels) if int(l) == label]
        subset1_index.extend(random.sample(class_samples, subset1_Num))
        subset2_index.extend(random.sample(class_samples, subset2_Num))

    subset1 = getDataset.GetSubset(dataset, subset1_index)
    subset2 = getDataset.GetSubset(dataset, subset2_index)

    return subset1, subset2


def calcuMetric(class_outputs, labels):

    TP = 0
    FN = 0
    TN = 0
    FP = 0

    for i in range(labels.size(0)):
        if labels[i] == 0 and class_outputs[i][0] > class_outputs[i][1]:
            TP = TP + 1
        elif labels[i] == 0 and class_outputs[i][0] < class_outputs[i][1]:
            FN = FN + 1
        elif labels[i] == 1 and class_outputs[i][0] < class_outputs[i][1]:
            TN = TN + 1
        elif labels[i] == 1 and class_outputs[i][0] > class_outputs[i][1]:
            FP = FP + 1

    return TP, FN, TN, FP

def testModel(testDataset, model, UserNum, batchSize, legalUser):

    testSet1 = reDataset(testDataset, legalUser, UserNum=UserNum)
    testSet1_loader = torch.utils.data.DataLoader(testSet1, batch_size=batchSize, shuffle=True, **kwargs)
    TP_all = 0
    FN_all = 0
    TN_all = 0
    FP_all = 0
    for data in testSet1_loader:
        samples, labels = data
        samples = samples.cuda()
        class_outputs, domain_outputs = model(samples, 0)
        TP, FN, TN, FP = calcuMetric(class_outputs, labels)
        TP_all = TP_all + TP
        FN_all = FN_all + FN
        TN_all = TN_all + TN
        FP_all = FP_all + FP
    FAR = FP_all / (FP_all + TN_all)
    FRR = FN_all / (TP_all + FN_all)
    TPR = TP_all / (TP_all + FN_all)
    TNR = TN_all / (TN_all + FP_all)
    BAC = 1 / 2 * (TPR + TNR)

    return FAR, FRR, TPR, TNR, BAC



if __name__ == '__main__':


    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    UserList = []
    for i in range(1,46):
        UserList.append(i)

    dataType1 = getDataset.getOriginDataset('/mainDataset', UserList=UserList, dataType=['chair_10'])
    dataType2 = getDataset.getOriginDataset('/mainDataset', UserList=UserList, dataType=['head_10'])
    dataType3 = getDataset.getOriginDataset('/mainDataset', UserList=UserList, dataType=['body_10'])
    dataType4 = getDataset.getOriginDataset('/mainDataset', UserList=UserList, dataType=['walk_10'])


    trainNum = 80

    trainType1, testType1 = get_Train_Test_Dataset(dataType1, UserList, trainNum=trainNum)
    trainType2, testType2 = get_Train_Test_Dataset(dataType2, UserList, trainNum=trainNum)
    trainType3, testType3 = get_Train_Test_Dataset(dataType3, UserList, trainNum=trainNum)
    trainType4, testType4 = get_Train_Test_Dataset(dataType4, UserList, trainNum=trainNum)

    n_epoch = 2
    batchSize = 8
    lr = 0.01
    legalUser_list = []
    for legalUser in UserList:
        legalUser_list.append(legalUser)

    FAR_totalMean = 0
    FRR_totalMean = 0
    TPR_totalMean = 0
    TNR_totalMean = 0
    BAC_totalMean = 0

    for legalUser in legalUser_list:

        print("batchSize, lr, n_epoch, trainNum, legalUser:", batchSize, lr, n_epoch, trainNum, legalUser)

        dataset1 = reDataset(trainType1, legalUser, UserNum=len(UserList))
        dataset2 = reDataset(trainType2, legalUser, UserNum=len(UserList))
        dataset3 = reDataset(trainType3, legalUser, UserNum=len(UserList))
        dataset4 = reDataset(trainType4, legalUser, UserNum=len(UserList))

        dataType1_loader = torch.utils.data.DataLoader(dataset1, batch_size=batchSize, shuffle=True, **kwargs)
        dataType2_loader = torch.utils.data.DataLoader(dataset2, batch_size=batchSize, shuffle=True, **kwargs)
        dataType3_loader = torch.utils.data.DataLoader(dataset3, batch_size=batchSize, shuffle=True, **kwargs)
        dataType4_loader = torch.utils.data.DataLoader(dataset4, batch_size=batchSize, shuffle=True, **kwargs)

        # load model
        my_net = BANNModel()

        if cuda:
            my_net.cuda()

        # setup optimizer
        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)

        loss_class = torch.nn.NLLLoss()
        loss_domain = torch.nn.NLLLoss()

        # training
        for epoch in range(n_epoch):

            scheduler.step()
            my_net.train()

            len_dataloader = min(len(dataType1_loader), len(dataType2_loader), len(dataType3_loader), len(dataType4_loader))  # 返回数据集中批次的数量
            datatype1_iter = iter(dataType1_loader)
            datatype2_iter = iter(dataType2_loader)
            datatype3_iter = iter(dataType3_loader)
            datatype4_iter = iter(dataType4_loader)

            lossTotal = 0
            Loss_class_loader = 0
            Loss_domain_loader = 0

            for i in range(len_dataloader):
                p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
                lamuda = 2. / (1. + np.exp(-10 * p)) - 1
                optimizer.zero_grad()

                data1 = next(datatype1_iter)
                loss1_label, loss1_domain = trainer(my_net, data1, data_domain=0)

                data2 = next(datatype2_iter)
                loss2_label, loss2_domain = trainer(my_net, data2, data_domain=1)

                data3 = next(datatype3_iter)
                loss3_label, loss3_domain = trainer(my_net, data3, data_domain=2)

                data4 = next(datatype4_iter)
                loss4_label, loss4_domain = trainer(my_net, data4, data_domain=3)

                lossTotal = loss1_label + loss1_domain + loss2_label + loss2_domain + loss3_label + loss3_domain + loss4_label + loss4_domain

                lossTotal.backward()
                optimizer.step()

        #model save
        # torch.save(my_net, "models/" + str(legalUser) + "_cae_attention_network10.pth")

        # testing
        my_net.eval()
        FAR_mean = 0
        FRR_mean = 0
        TPR_mean = 0
        TNR_mean = 0
        BAC_mean = 0
        with torch.no_grad():

            FAR1, FRR1, TPR1, TNR1, BAC1 = testModel(testType1, my_net, len(UserList), batchSize, legalUser)
            print("chair_10, FAR1, FRR1, TPR1, TNR1, BAC1:", FAR1, FRR1, TPR1, TNR1, BAC1)

            FAR2, FRR2, TPR2, TNR2, BAC2 = testModel(testType2, my_net, len(UserList), batchSize, legalUser)
            print("head_10, FAR2, FRR2, TPR2, TNR2, BAC2:" , FAR2, FRR2, TPR2, TNR2, BAC2)

            FAR3, FRR3, TPR3, TNR3, BAC3 = testModel(testType3, my_net, len(UserList), batchSize, legalUser)
            print("body_10, FAR3, FRR3, TPR3, TNR3, BAC3:",  FAR3, FRR3, TPR3, TNR3, BAC3)

            FAR4, FRR4, TPR4, TNR4, BAC4 = testModel(testType4, my_net, len(UserList), batchSize, legalUser)
            print("walk_10, FAR4, FRR4, TPR4, TNR4, BAC4:",  FAR4, FRR4, TPR4, TNR4, BAC4)

            FAR_mean = FAR_mean + FAR1 + FAR2 + FAR3 + FAR4
            FRR_mean = FRR_mean + FRR1 + FRR2 + FRR3 + FRR4
            TPR_mean = TPR_mean + TPR1 + TPR2 + TPR3 + TPR4
            TNR_mean = TNR_mean + TNR1 + TNR2 + TNR3 + TNR4
            BAC_mean = BAC_mean + BAC1 + BAC2 + BAC3 + BAC4
        print("FAR_mean, FRR_mean, TPR_mean, TNR_mean, BAC_mean:", FAR_mean/4, FRR_mean/4, TPR_mean/4, TNR_mean/4, BAC_mean/4)

        FAR_totalMean = FAR_totalMean + FAR_mean/4
        FRR_totalMean = FRR_totalMean + FRR_mean/4
        TPR_totalMean = TPR_totalMean + TPR_mean/4
        TNR_totalMean = TNR_totalMean + TNR_mean/4
        BAC_totalMean = BAC_totalMean + BAC_mean/4
    print("FAR_totalMean, FRR_totalMean, TPR_totalMean, TNR_totalMean, BAC_totalMean:",
          FAR_totalMean / 5, FRR_totalMean / 5, TPR_totalMean / 5, TNR_totalMean / 5, BAC_totalMean / 5)


