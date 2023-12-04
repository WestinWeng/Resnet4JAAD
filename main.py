import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import JAADDataset
import random
from model import ResNet50,ResNet101,ResNet152
import torch.nn.functional as F
import numpy as np

#hyper parameter
backbone_list=['ResNet50', 'ResNet101','ResNet152']
backbone='ResNet50'
batch_size=64
epoch_number=50
lr=0.001
device=('cuda:0' if torch.cuda.is_available() else 'cpu' )
weight_decay=8e-5
save_dir='save.npz'

#train dir
root_dir='/root/autodl-tmp/data/'
train_scene_dir='train_scene/'
train_ped_dir='train_ped/'
train_label='train_label.txt'

#test_dir
test_scene_dir='test_scene/'
test_ped_dir='test_ped/'
test_label='test_label.txt'

#dataset
train_set=JAADDataset(root_dir,train_scene_dir,train_ped_dir,train_label)
test_set=JAADDataset(root_dir,test_scene_dir,test_ped_dir,test_label)

#dataloader
train_dataloader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_set,batch_size=batch_size,shuffle=False)

#exp implementation
get_model=[ResNet50,ResNet101,ResNet152]
model=get_model[backbone_list.index(backbone)]()
model=model.to(device)
criterion=nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True,
#                                                           threshold=0.0001,
#                                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
train_loss_list=[]
train_accu_list=[]
test_loss_list=[]
test_accu_list=[]
high_train_accu=0.0
high_test_accu=0.0
for epoch in range(0,epoch_number):
    train_loss=0.0
    train_accu=0.0
    deno=len(train_dataloader)*1.0
    for iter,(scene,pedestrain,label) in enumerate(train_dataloader):
        temp_accu=0.0
        model.train()
        scene=scene.to(device)
        pedestrain=pedestrain.to(device)
        label=label.to(device)
        predic=model(scene,pedestrain)
        loss=criterion(predic,label)
        _,pred_label=torch.max(F.softmax(predic),dim=1)
        temp_accu+=sum([1 for x1,x2 in zip(pred_label, label) if x1==x2])


        #upgrade
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())
        train_accu += temp_accu
    train_accu=train_accu/(deno*batch_size*1.0)
    train_loss=train_loss/(deno*1.0)


    #test:
    test_loss=0.0
    test_accu=0.0
    t_deno=len(test_dataloader)*1.0
    for iter,(t_scene,t_pedestrain,t_label) in enumerate(test_dataloader):
        model.eval()
        t_scene=t_scene.to(device)
        t_pedestrain=t_pedestrain.to(device)
        t_label=t_label.to(device)
        t_predic=model(t_scene,t_pedestrain)
        loss=criterion(predic,label)
        _,t_pred_label=torch.max(F.softmax(predic),dim=1)
        test_accu+=sum([1 for x1,x2 in zip(t_pred_label, t_label) if x1==x2])
        test_loss+=float(loss.item())
    test_loss=test_loss/(t_deno)
    test_accu=test_accu/(t_deno*batch_size*1.0)

    if train_accu >= high_train_accu:
        high_train_accu=train_accu
    if test_accu>= high_test_accu:
        high_test_accu=test_accu

    print(f'Epoch {epoch} \n'
              f'train Loss[{train_loss:.4f}] train_accuracy[{train_accu:.4f}] \n'
              f'test Loss[{test_loss:.4f}] test_accuracy[{test_accu:.4f}] \n'
              f'high train acc: {high_train_accu:.4f} high test acc: {high_test_accu:.4f}\n')
    
    train_accu_list.append(train_accu)
    test_accu_list.append(test_accu)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

np.savez(save_dir,
    train_accuracy = train_accu_list,
    test_accuracy=test_accu_list,
    train_loss=train_loss_list,
    test_loss=test_loss_list)
        


        









        



        
