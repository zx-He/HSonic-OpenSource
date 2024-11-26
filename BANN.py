
import getDataset
from torch.utils.data import Dataset
import torch
from model import BANNModel
import torch.optim as optim
import numpy as np
from torch.optim import lr_scheduler


def trainer(my_net, data_source):
    data, class_label, domain_label = data_source

    my_net.zero_grad()

    if cuda:
        data = data.cuda()
        class_label = class_label.cuda()
        domain_label = domain_label.cuda()

    class_output, domain_output = my_net(input_data=data, lamuda=lamuda)

    loss_s_class = loss_class(class_output, class_label)
    loss_s_domain = loss_domain(domain_output, domain_label)

    return loss_s_class, loss_s_domain



if __name__ == '__main__':


    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


    defaultList = []
    for i in range(1, 31):
        defaultList.append(i)


    dataType1 = getDataset.getDatasetwithDoamin('/hybridFeatures/', UserList=defaultList, dataType=['chair_10'], domainLable=0)
    dataType2 = getDataset.getDatasetwithDoamin('/hybridFeatures/', UserList=defaultList, dataType=['head_10'], domainLable=1)
    dataType3 = getDataset.getDatasetwithDoamin('/hybridFeatures/', UserList=defaultList, dataType=['body_10'], domainLable=2)
    dataType4 = getDataset.getDatasetwithDoamin('/hybridFeatures/', UserList=defaultList, dataType=['walk_10'], domainLable=3)

    n_epoch = 60
    batchSize = 8
    lr = 0.01

    dataType1_loader = torch.utils.data.DataLoader(dataType1, batch_size=batchSize, shuffle=True, **kwargs)
    dataType2_loader = torch.utils.data.DataLoader(dataType2, batch_size=batchSize, shuffle=True, **kwargs)
    dataType3_loader = torch.utils.data.DataLoader(dataType3, batch_size=batchSize, shuffle=True, **kwargs)
    dataType4_loader = torch.utils.data.DataLoader(dataType4, batch_size=batchSize, shuffle=True, **kwargs)


    legalUser_list = []
    for legalUser in defaultList:
        legalUser_list.append(legalUser)

    FAR_totalMean = 0
    FRR_totalMean = 0
    TPR_totalMean = 0
    TNR_totalMean = 0
    BAC_totalMean = 0

    my_net = BANNModel()

    if cuda:
        my_net.cuda()

    optimizer = optim.Adam(my_net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, last_epoch=-1)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    for epoch in range(n_epoch):

        scheduler.step()
        my_net.train()

        len_dataloader = min(len(dataType1_loader), len(dataType2_loader), len(dataType3_loader),
                             len(dataType4_loader))
        datatype1_iter = iter(dataType1_loader)
        datatype2_iter = iter(dataType2_loader)
        datatype3_iter = iter(dataType3_loader)
        datatype4_iter = iter(dataType4_loader)

        lossTotal = 0


        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            lamuda = 2. / (1. + np.exp(-10 * p)) - 1
            optimizer.zero_grad()

            data1 = next(datatype1_iter)
            loss1_label, loss1_domain = trainer(my_net, data1)

            data2 = next(datatype2_iter)
            loss2_label, loss2_domain = trainer(my_net, data2)

            data3 = next(datatype3_iter)
            loss3_label, loss3_domain = trainer(my_net, data3)

            data4 = next(datatype4_iter)
            loss4_label, loss4_domain = trainer(my_net, data4)

            lossTotal = loss1_label + loss1_domain + loss2_label + loss2_domain + loss3_label + loss3_domain + loss4_label + loss4_domain

            lossTotal.backward()
            optimizer.step()

            print("epoch, loss_label, loss1_domain:", epoch, loss1_label+loss2_label+loss3_label+loss4_label,
                  loss1_domain + loss2_domain + loss3_domain + loss4_domain)

    torch.save(my_net.feature_extractor.state_dict(), 'FeatureExtractor.pth')
    torch.save(my_net.user_classifier.state_dict(), 'UserClassifier.pth')
    torch.save(my_net.behavior_classifier.state_dict(), 'BehaviorClassifier.pth')



