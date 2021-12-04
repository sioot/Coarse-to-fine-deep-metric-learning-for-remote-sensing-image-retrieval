from __future__ import print_function
from __future__ import division
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
#from utils import features_roi
#from retrieval_ import testing, cal_rating


import copy

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

data_dir = "C:\\Users\\yun\\Desktop\\데이터새로\\Train\\"
#data_dir2 = "D:\\새로운데이터\\성능평가\\통합\\0.6\\Retrieval\\"

#data_dir = "D:\\google_earth\\train_for_daejun\\for_model\\"
#save_dir = "C:\\Users\\yun\\Desktop\\zero_base\\"
model_name = "resnet_triple"

batch_size = 32

num_epochs = 50

feature_extract = False

data_transforms2 = {
    'DB': transforms.Compose([
        transforms.CenterCrop(800),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'query': transforms.Compose([
        transforms.CenterCrop(800),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# image_datasets2 = {x: datasets.ImageFolder(os.path.join(data_dir2, x), data_transforms2[x]) for x in ['DB', 'query']}
# dataloaders_dict2 = {
# x: torch.utils.data.DataLoader(image_datasets2[x], batch_size=batch_size, shuffle=False, num_workers=0) for x in
# ['DB', 'query']}



def normal(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)



def calculate_IOU(x,y, z,w):
    size = 864
    cross = ((864 - abs(x-z)) * (864- abs(y-w)))
    IOU = 1-(cross/ ((864 * 864)*2-cross))
    return IOU




class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)




class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


epoch_loss=[]
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    tr_epoch_loss = []
    te_loss = []
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 50)
        for phase in ['train', 'val']:
            margin_push = 0.1
            loss_dic = {'train': 0.0, 'val': 0.0}

            loss_dic[phase] = 0
            batch_loss = torch.tensor(0.0)
            batch_loss = batch_loss.to(device)
            if phase == 'train':
                model.train()  # Set model to training mode
                for i in range(len(dataloaders[phase].dataset)//2):
                    # print((batch_size*len(dataloaders[phase]) // 2))
                    optimizer.zero_grad()
                    img = Image.open(dataloaders[phase].dataset.samples[i][0])
                    # print(dataloaders[phase].dataset.samples[i][0])
                    anchor = dataloaders[phase].dataset.transform(img)
                    anchor = anchor.to(device)


                    img = Image.open(dataloaders[phase].dataset.samples[i + (len(dataloaders[phase].dataset) // 2)][0])
                    # print(dataloaders[phase].dataset.samples[ i + batch_size*(len(dataloaders[phase])//2) ][0])
                    positive = dataloaders[phase].dataset.transform(img)
                    positive = positive.to(device)

                    m = random.randint(1,5000)
                    if (i + m > len(dataloaders[phase].dataset) // 2):
                        img = Image.open(
                            dataloaders[phase].dataset.samples[i + (len(dataloaders[phase].dataset) // 2 - m)][0])

                    else:
                        img = Image.open(
                            dataloaders[phase].dataset.samples[(len(dataloaders[phase].dataset) // 2 + m)][0])
                    # print(dataloaders[phase].dataset.samples[(i+len(dataloaders[phase].dataset)//2 + 1)][0])
                    negative = dataloaders[phase].dataset.transform(img)
                    negative = negative.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        an_out = normal(model(anchor.unsqueeze(0)))
                        pos_out = normal(model(positive.unsqueeze(0)))
                        neg_out = normal(model(negative.unsqueeze(0)))
                        # for i in range (16):
                        #     m = random.randint(1, 5000)
                        #     img1 = Image.open(
                        #         dataloaders[phase].dataset.samples[(len(dataloaders[phase].dataset) // 2 + m)][0])
                        #     negative1 = dataloaders[phase].dataset.transform(img1)
                        #     negative1 = negative1.to(device)
                        #     neg_out1 = normal(model(negative1.unsqueeze(0)))
                        #     distance_negative_1 = (an_out[0] - neg_out1[0]).pow(2).sum(0)
                        #     distance_negative = (an_out[0] - neg_out[0]).pow(2).sum(0)  # .pow(.5)
                        #     if distance_negative_1<distance_negative:
                        #         neg_out = neg_out1


                        loss = criterion(an_out, pos_out, neg_out, margin_push)
                        batch_loss = loss + batch_loss
                        # backward + optimize only if in training phase
                        if (i + 1) % (batch_size) == 0:
                            if batch_loss == 0:
                                batch_loss = torch.tensor(0.0)
                                batch_loss = batch_loss.to(device)
                                pass
                            else:
                                batch_loss.backward()
                                optimizer.step()
                                tr_epoch_loss.append(np.float(batch_loss))
                                batch_loss = torch.tensor(0.0)
                                batch_loss = batch_loss.to(device)

                    loss_dic[phase] = torch.sum(loss) + loss_dic[phase]

                epoch_loss = loss_dic[phase] / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                # torch.save(model.state_dict(), 'C:\\Users\\yun\\Desktop\\zone\\zero_base\\{}.pt'.format(epoch))
                torch.save(model.state_dict(), 'C:\\Users\yun\\Desktop\\small\\800\\without_cross\\{}.pt'.format(epoch))


            if phase == 'val':
                for i in range(len(dataloaders[phase].dataset) // 2):
                    # print((batch_size*len(dataloaders[phase]) // 2))
                    optimizer.zero_grad()
                    img = Image.open(dataloaders[phase].dataset.samples[i][0])
                    # print(dataloaders[phase].dataset.samples[i][0])
                    anchor = dataloaders[phase].dataset.transform(img)
                    anchor = anchor.to(device)

                    img = Image.open(dataloaders[phase].dataset.samples[i + (len(dataloaders[phase].dataset) // 2)][0])
                    # print(dataloaders[phase].dataset.samples[ i + batch_size*(len(dataloaders[phase])//2) ][0])
                    positive = dataloaders[phase].dataset.transform(img)
                    positive = positive.to(device)

                    if (i + 1 > len(dataloaders[phase].dataset) // 2):
                        img = Image.open(
                            dataloaders[phase].dataset.samples[i + (len(dataloaders[phase].dataset) // 2 - 1)][0])
                    else:
                        img = Image.open(
                            dataloaders[phase].dataset.samples[(len(dataloaders[phase].dataset) // 2 + 1)][0])
                    # print(dataloaders[phase].dataset.samples[(i+len(dataloaders[phase].dataset)//2 + 1)][0])
                    negative = dataloaders[phase].dataset.transform(img)
                    negative = negative.to(device)

                    with torch.no_grad():
                        an_out = normal(model(anchor.unsqueeze(0)))
                        pos_out = normal(model(positive.unsqueeze(0)))
                        neg_out = normal(model(negative.unsqueeze(0)))


                        loss = criterion(an_out, pos_out, neg_out, margin_push)
                        batch_loss = loss + batch_loss
                        # backward + optimize only if in training phase
                        if (i + 1) % (batch_size) == 0:
                            te_loss.append(np.float(batch_loss))
                            batch_loss = torch.tensor(0.0)
                            batch_loss = batch_loss.to(device)
                    # DB_name, DB_feature, query_name, query_feature = testing(model, dataloaders2)
                    # rating = cal_rating(DB_name, DB_feature, query_name, query_feature)
                    loss_dic[phase] = torch.sum(loss) + loss_dic[phase]

                    epoch_loss = loss_dic[phase] / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print()


            #print(dataloaders['train'].dataset.samples[3999][0])
            #print(dataloaders['train'].dataset.samples[4000][0])
            #print(len(dataloaders['train'].dataset) // 2)
            #print(len(dataloaders['val'].dataset) // 2)


            # Iterate over data.

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, tr_epoch_loss, te_loss

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, feature_extract, use_pretrained): #True
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name =="resnet_triple":
        model_ft = models.resnet34(pretrained=True)
        model_ft.fc = nn.Linear(512, 512)
        modules = list(model_ft.children())
        modules = modules[:-2]
        model_ft = nn.Sequential(*modules)
        model_ft.add_module('1*1_averagepooling', nn.AvgPool3d(kernel_size=(8, 1, 1), stride=(8, 1, 1)))
        model_ft.Flat = Flatten()
        model_ft.fc = nn.Linear(3136, 512)
        # checkpoint = torch.load('C:\\Users\yun\\Desktop\\small\\800\\Triplet Loss Models\\128\\49.pt')
        # model_ft.load_state_dict(checkpoint)
        input_size = 224

    elif model_name == "Densenet161":
        """ resnet + fc
        """
        model_ft = models.densenet161(pretrained=True)
        modules = list(model_ft.children())[:-1]
        modules = modules[0][:-2][:-1]
        model_ft = nn.Sequential(*modules)
        #checkpoint = torch.load('./AID_fine/6000_auto/109.pt')
        #model_ft.load_state_dict(checkpoint)
        input_size = 224

# output : [1, 2254]
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, positive, negative, margin_push, size_average=True):
        distance_positive = (anchor[0] - positive[0]).pow(2).sum(0)  # .pow(.5)
        distance_negative = (anchor[0] - negative[0]).pow(2).sum(0)  # .pow(.5)
        loss_n1 = torch.max(torch.tensor([0, margin_push-distance_negative+distance_positive],requires_grad=True)).cuda() + distance_positive
        return loss_n1.mean()


loss = TripletLoss()
criterion = loss


model_ft, input_size = initialize_model(model_name,feature_extract, use_pretrained=True)

print(model_ft) 

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop(800),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(800),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=0) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()


print("Params to learn:")

if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer_ft = optim.SGD(params_to_update, lr=0.0001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

model_ft, trlosshistory,telosshistory = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
# np.save('./triplet/train_his.npy',trlosshistory)
# np.save('./triplet/test_his.npy',telosshistory)
# #np.save('./scratch_tr_history',trlosshistory)
# #np.save('./scratch_te_history',telosshistory)
