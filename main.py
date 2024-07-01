from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import mean_absolute_error as mae
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import *
from model import *
from dataset import *
import os
import copy
import config
import numpy as np
import torch
import torch.nn as nn
import csv
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
import warnings
warnings.filterwarnings("ignore")

def train():
    mse = nn.MSELoss()
    diffusion = Diffusion()
    average_now = 0

    text_encoder = Text_encoder().to(config.device)
    Unet = UNet().to(config.device)
    opt_model = optim.AdamW(list(Unet.parameters())+list(text_encoder.parameters()), lr=config.learning_rate)
    ema = EMA(0.9999)
    ema_Unet = copy.deepcopy(Unet).eval().requires_grad_(False)
    data = pd.read_csv("../data_info/info.csv",encoding = "ISO-8859-1")

    for epoch in range(config.epochs):
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","MSE_loss", "Intra_domain_loss"])
        
        dataset = OneDataset(root_MRI = config.whole_MRI, task = config.train, name = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.__len__()
        MSE_loss_epoch = 0
        Intra_domain_loss_epoch = 0

        for idx, (MRI, MRI_name) in enumerate(loop):
            Abeta_path = config.whole_Abeta+"/"+MRI_name[0]
            Tau_path = config.whole_Tau+"/"+MRI_name[0]

            if os.path.exists(Abeta_path) or os.path.exists(Tau_path):
                MRI = np.expand_dims(MRI, axis=1)
                MRI = torch.tensor(MRI)
                MRI = MRI.to(config.device)

                Age = data[data['ID'] == MRI_name[0][0:-4]]['Age']
                Age = Age.values
                Age = Age.astype(np.float32)
                Sex = data[data['ID'] == MRI_name[0][0:-4]]['Sex']
                Sex = Sex.values

                t = diffusion.sample_timesteps(MRI.shape[0]).to(config.device)

                if os.path.exists(Abeta_path):
                    Abeta_text = f"Synthesize an Aβ-PET scan for a {Age[0]}-year-old {Sex[0]} subject"
                    Abeta_text_feature = text_encoder(clip.tokenize(Abeta_text).to(config.device))
                    Abeta = nifti_to_numpy(Abeta_path)
                    Abeta = np.expand_dims(Abeta, axis=0)
                    Abeta = np.expand_dims(Abeta, axis=1)
                    Abeta = torch.tensor(Abeta)
                    Abeta = Abeta.to(config.device)
                    Abeta_t = diffusion.noise_images(Abeta, MRI, t)
                    Abeta_out = Unet(Abeta_t, t, Abeta_text_feature)
                else:
                    Abeta_out = None

                if os.path.exists(Tau_path):
                    Tau_text = f"Synthesize an Tau-PET scan for a {Age[0]}-year-old {Sex[0]} subject"
                    Tau_text_feature = text_encoder(clip.tokenize(Tau_text).to(config.device))
                    Tau = nifti_to_numpy(Tau_path)
                    Tau = np.expand_dims(Tau, axis=0)
                    Tau = np.expand_dims(Tau, axis=1)
                    Tau = torch.tensor(Tau)
                    Tau = Tau.to(config.device)
                    Tau_t = diffusion.noise_images(Tau, MRI, t)
                    Tau_out = Unet(Tau_t, t, Tau_text_feature)
                else:
                    Tau_out = None

                MSE_loss = 0
                opt_model.zero_grad()

                if Abeta_out is not None:
                    MSE_loss += mse(Abeta_out, Abeta)
                if Tau_out is not None:
                    MSE_loss += mse(Tau_out, Tau) 

                if (Abeta_out is not None) and (Tau_out is not None):
                    Intra_domain_loss = mse(torch.abs(Abeta-Tau),torch.abs(Abeta_out-Tau_out))

                loss = MSE_loss * 10 + Intra_domain_loss
                opt_model.zero_grad()
                loss.backward()
                opt_model.step()
                ema.step_ema(ema_Unet, Unet)

                MSE_loss_epoch = MSE_loss_epoch + MSE_loss
                Intra_domain_loss_epoch = Intra_domain_loss_epoch + Intra_domain_loss

        writer.writerow([epoch+1, MSE_loss_epoch.item()/length, MSE_loss_epoch.item()/length])
        lossfile.close()

        #validation
        dataset = OneDataset(root_MRI = config.whole_MRI, task = config.validation, name = "validation")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.__len__()

        mae_0_Abeta = 0
        psnr_0_Abeta = 0
        ssim_0_Abeta = 0
        mae_0_Tau = 0
        psnr_0_Tau = 0
        ssim_0_Tau = 0
        count_Abeta = 0
        count_Tau = 0

        for idx, (MRI, MRI_name) in enumerate(loop):
            validation_file = open("result/"+str(config.exp)+"validation.csv", 'a+', newline = '')
            writer = csv.writer(validation_file)
            if epoch == 0 and idx == 0:
                writer.writerow(["Epoch","Name","PSNR","SSIM"])

            MRI = np.expand_dims(MRI, axis=1)
            MRI = torch.tensor(MRI)
            MRI = MRI.to(config.device)

            Age = data[data['ID'] == MRI_name[0][0:-4]]['Age']
            Age = Age.values
            Age = Age.astype(np.float32)
            Sex = data[data['ID'] == MRI_name[0][0:-4]]['Sex']
            Sex = Sex.values

            t = diffusion.sample_timesteps(MRI.shape[0]).to(config.device)

            Abeta_path = config.whole_Abeta+"/"+MRI_name[0]
            if os.path.exists(Abeta_path):
                Abeta_text = f"Synthesize an Aβ-PET scan for a {Age[0]}-year-old {Sex[0]} subject"
                Abeta_text_feature = text_encoder(clip.tokenize(Abeta_text).to(config.device))
                count_Abeta += 1
                Abeta = nifti_to_numpy(Abeta_path)
                Abeta_out = diffusion.sample(ema_Unet, MRI, Abeta_text_feature)
                Abeta_out = torch.clamp(Abeta_out,0,1)
                Abeta_out = Abeta_out.detach().cpu().numpy()
                Abeta_out = np.squeeze(Abeta_out)
                Abeta_out = Abeta_out.astype(np.float32)

                fake_PET_flatten = Abeta_out.reshape(-1,128)
                True_PET_flatten = Abeta.reshape(-1,128)
                mae_0_Abeta += mae(True_PET_flatten,fake_PET_flatten)
                psnr_0_Abeta += round(psnr(Abeta,Abeta_out),3)
                ssim_0_Abeta += round(ssim(Abeta,Abeta_out),3)

            Tau_path = config.whole_Tau+"/"+MRI_name[0]
            if os.path.exists(Tau_path):
                Tau_text = f"Synthesize an Tau-PET scan for a {Age[0]}-year-old {Sex[0]} subject"
                Tau_text_feature = text_encoder(clip.tokenize(Tau_text).to(config.device))
                count_Tau += 1
                Tau = nifti_to_numpy(Tau_path)
                Tau_out = diffusion.sample(ema_Unet, MRI,Tau_text_feature)
                Tau_out = torch.clamp(Tau_out,0,1)
                Tau_out = Tau_out.detach().cpu().numpy()
                Tau_out = np.squeeze(Tau_out)
                Tau_out = Tau_out.astype(np.float32)

                fake_PET_flatten = Tau_out.reshape(-1,128)
                True_PET_flatten = Tau.reshape(-1,128)
                mae_0_Tau += mae(True_PET_flatten,fake_PET_flatten)
                psnr_0_Tau += round(psnr(Tau,Tau_out),3)
                ssim_0_Tau += round(ssim(Tau,Tau_out),3)

        writer.writerow([epoch,mae_0_Abeta/count_Abeta,psnr_0_Abeta/count_Abeta,ssim_0_Abeta/count_Abeta])
        writer.writerow([epoch,mae_0_Tau/count_Tau,psnr_0_Tau/count_Tau,ssim_0_Tau/count_Tau])
        validation_file.close()

        #test
        average = psnr_0_Abeta/count_Abeta + ssim_0_Abeta/count_Abeta * 10 + psnr_0_Tau/count_Tau + ssim_0_Tau/count_Tau * 10
        if average > average_now:
            average_now = average
            save_checkpoint(ema_Unet,opt_model,filename=config.CHECKPOINT_Unet)
            save_checkpoint(text_encoder,opt_model,filename=config.CHECKPOINT_encoder)

            dataset = OneDataset(root_MRI = config.whole_MRI, task = config.test, name = "test")
            loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)
            length = dataset.__len__()

            mae_0_Abeta = 0
            psnr_0_Abeta = 0
            ssim_0_Abeta = 0
            mae_0_Tau = 0
            psnr_0_Tau = 0
            ssim_0_Tau = 0
            count_Abeta = 0
            count_Tau = 0

            for idx, (MRI, MRI_name) in enumerate(loop):
                test_file = open("result/"+str(config.exp)+"test.csv", 'a+', newline = '')
                writer = csv.writer(test_file)
                if epoch == 0 and idx == 0:
                    writer.writerow(["Epoch","Name","PSNR","SSIM"])

                MRI = np.expand_dims(MRI, axis=1)
                MRI = torch.tensor(MRI)
                MRI = MRI.to(config.device)

                Age = data[data['ID'] == MRI_name[0][0:-4]]['Age']
                Age = Age.values
                Age = Age.astype(np.float32)
                Sex = data[data['ID'] == MRI_name[0][0:-4]]['Sex']
                Sex = Sex.values

                t = diffusion.sample_timesteps(MRI.shape[0]).to(config.device)

                Abeta_path = config.whole_Abeta+"/"+MRI_name[0]
                if os.path.exists(Abeta_path):
                    Abeta_text = f"Synthesize an Aβ-PET scan for a {Age[0]}-year-old {Sex[0]} subject"
                    Abeta_text_feature = text_encoder(clip.tokenize(Abeta_text).to(config.device))
                    count_Abeta += 1
                    Abeta = nifti_to_numpy(Abeta_path)
                    Abeta_out = diffusion.sample(ema_Unet, MRI, Abeta_text_feature)
                    Abeta_out = torch.clamp(Abeta_out,0,1)
                    Abeta_out = Abeta_out.detach().cpu().numpy()
                    Abeta_out = np.squeeze(Abeta_out)
                    Abeta_out = Abeta_out.astype(np.float32)

                    fake_PET_flatten = Abeta_out.reshape(-1,128)
                    True_PET_flatten = Abeta.reshape(-1,128)
                    mae_0_Abeta += mae(True_PET_flatten,fake_PET_flatten)
                    psnr_0_Abeta += round(psnr(Abeta,Abeta_out),3)
                    ssim_0_Abeta += round(ssim(Abeta,Abeta_out),3)

                Tau_path = config.whole_Tau+"/"+MRI_name[0]
                if os.path.exists(Tau_path):
                    Tau_text = f"Synthesize an Tau-PET scan for a {Age[0]}-year-old {Sex[0]} subject"
                    Tau_text_feature = text_encoder(clip.tokenize(Tau_text).to(config.device))
                    count_Tau += 1
                    Tau = nifti_to_numpy(Tau_path)
                    Tau_out = diffusion.sample(ema_Unet, MRI,Tau_text_feature)
                    Tau_out = torch.clamp(Tau_out,0,1)
                    Tau_out = Tau_out.detach().cpu().numpy()
                    Tau_out = np.squeeze(Tau_out)
                    Tau_out = Tau_out.astype(np.float32)

                    fake_PET_flatten = Tau_out.reshape(-1,128)
                    True_PET_flatten = Tau.reshape(-1,128)
                    mae_0_Tau += mae(True_PET_flatten,fake_PET_flatten)
                    psnr_0_Tau += round(psnr(Tau,Tau_out),3)
                    ssim_0_Tau += round(ssim(Tau,Tau_out),3)

            writer.writerow([epoch,mae_0_Abeta/count_Abeta,psnr_0_Abeta/count_Abeta,ssim_0_Abeta/count_Abeta])
            writer.writerow([epoch,mae_0_Tau/count_Tau,psnr_0_Tau/count_Tau,ssim_0_Tau/count_Tau])
            test_file.close()

if __name__ == '__main__':
    seed_torch()
    train()
