import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.init
import torchvision.utils as vutils
import numpy as np
from datasetloader import GOPRODataset
from model import MainModel
import tensorflow as tf
import datetime
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, loss_fn, loader, optimizer=None, trainmode=True):
    if trainmode:
        model.train()
    else:
        model.eval()
    step = 0
    total_loss = 0
    total_len = 0
    pbar = tqdm(loader, total=len(loader))
    
    for data in pbar:
        blur_image1, sharp_image1 = data
        device = torch.device("cuda")
        blur_image1 = blur_image1.to(device)
        sharp_image1 = sharp_image1.to(device)
        blur_image2 = F.interpolate(blur_image1, scale_factor=0.5, mode="bicubic")
        blur_image3 = F.interpolate(blur_image2, scale_factor=0.5, mode="bicubic")
        sharp_image2 = F.interpolate(sharp_image1, scale_factor=0.5, mode="bicubic")
        sharp_image3 = F.interpolate(sharp_image2, scale_factor=0.5, mode="bicubic")
        
        if trainmode:
            optimizer.zero_grad()
            model_out3, model_out2, model_out1 = model(blur_image3, blur_image2, blur_image1)
            loss3 = loss_fn(model_out3, sharp_image3)
            loss2 = loss_fn(model_out2, sharp_image2)
            loss1 = loss_fn(model_out1, sharp_image1)
            loss = loss1+loss2+loss3
            loss.backward()
            optimizer.step()
            tlosses.append(loss.item())
            
        else:
            with torch.no_grad():
                model_out3, model_out2, model_out1 = model(blur_image3, blur_image2, blur_image1)
                loss3 = loss_fn(model_out3, sharp_image3)
                loss2 = loss_fn(model_out2, sharp_image2)
                loss1 = loss_fn(model_out1, sharp_image1)
                loss = loss1+loss2+loss3
                vlosses.append(loss.item())
                
        total_loss += loss.item()
        total_len += blur_image1.shape[0]
                
        step += 1
        pbar.set_postfix(loss=total_loss/total_len)
        
    return total_loss/total_len



if __name__ =='__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    logdir = "logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter()

    train_transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.RandomCrop([256, 256]),
                                                    transforms.Compose([
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomApply(
                                                    [transforms.RandomRotation(degrees=90)], p=0.8),
                                                    #transforms.RandomApply(
                                                    #[transforms.ColorJitter(saturation=(0.5, 1.5))], p=0.6), 
                                                    #transforms.RandomApply(
                                                    #[transforms.GaussianBlur(5, (0.1, 2.0))],p=0.5),
                                                    ]),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    training_set = GOPRODataset(
        train_path='./data/GOPRO_Large/train',
        train_ext='png',
        transform=train_transform
    )

    test_set = GOPRODataset(
        train_path='./data/GOPRO_Large/test',
        train_ext='png',
        transform=test_transform
    )

    
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    tlosses = []
    vlosses = []
    
    epoch = 100
    log_interval = 5
    model = MainModel(3, 64, 128)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model = model.to("cuda")
    
    #loader
    """ model_path = './models_4'
    load_path = '{}/trained_model_epoch{}.pth'.format(model_path, 1)
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    print('Model {} loaded!'.format(load_path)) """


    for ep in range(epoch):
        tloss = train_one_epoch(model, loss_fn, training_loader, optimizer, trainmode=True)
        print('Epoch', '{:d} completed.'.format(ep+1), 'Train loss', '{:.3f}'.format(tloss))
        writer.add_scalar('Loss/Train', tloss, ep+1)

        """ vloss = train_one_epoch(model, loss_fn, test_loader, optimizer, trainmode=False)
        print('Epoch', '{:d} completed.'.format(ep+1), 'Validation loss', '{:.3f}'.format(vloss)) """
        
        model_path = './models_5'
        save_path_t = '{}/tlosses_epoch{}.npy'.format(model_path, ep+1)
        model_save_path = '{}/trained_model_epoch{}.pth'.format(model_path, ep+1)

        #save_path_v = '{}/vlosses_epoch{}.npy'.format(model_path, ep+1)

        #if ep+1 % log_interval == 0:
        np.save(save_path_t, np.array(tlosses))
        #np.save(save_path_v, np.array(vlosses))
        torch.save(model.state_dict(), model_save_path)

        