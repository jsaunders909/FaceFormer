import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from Discriminator import Discriminator


from data_loader import get_dataloaders
from faceformer_GAN import FaceformerGAN

def trainer(args, train_loader, dev_loader, model, D, G_optimizer, D_optimizer, criterion, epoch=100):
    save_path = os.path.join(args.dataset,args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    for e in range(epoch+1):
        G_loss_log, D_loss_log, recon_log, G_GAN_log, D_GAN_real_log, D_GAN_fake_log = [], [], [], [], [], []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader), position=0, leave=True)
        G_optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")

            # -------------- Discriminator ---------------
            D_optimizer.zero_grad()
            D_loss, D_loss_real, D_loss_fake = model.forward_D(audio, template, vertice, criterion,
                                                               one_hot, D, teacher_forcing=False)
            if D_loss > 0.05 * args.w_GAN:
                D_loss.backward()
            else:
                print('D not training')
            D_optimizer.step()

            # -------------- Generator -------------------
            G_optimizer.zero_grad()
            G_loss, recon_loss, G_loss_GAN = model.forward_G(audio, template, vertice, criterion,
                                                             one_hot, D, teacher_forcing=False)
            if G_loss > 0.05 * args.w_GAN:
                G_loss.backward()
            else:
                print('G not training')
            G_optimizer.step()

            G_loss_log.append(G_loss.item())
            D_loss_log.append(D_loss.item())
            recon_log.append(recon_loss.item())
            G_GAN_log.append(G_loss_GAN.item())
            D_GAN_real_log.append(D_loss_real.item())
            D_GAN_fake_log.append(D_loss_fake.item())

            for log in [G_loss_log, D_loss_log, recon_log, G_GAN_log, D_GAN_real_log, D_GAN_fake_log]:
                if len(log) > 50:
                    log.pop(0)

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS RECON:{:.7f} G GAN:{:.5f} D GAN REAL:{:.3f} D GAN FAKE:{:.3f} D LOSS:{:.5f}".format((e+1),
                   iteration, np.mean(recon_log), np.mean(G_GAN_log), np.mean(D_GAN_real_log), np.mean(D_GAN_fake_log), (np.mean(D_GAN_real_log) + np.mean(D_GAN_fake_log)) /2))
        # validation
        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all,file_name in dev_loader:
            # to gpu
            audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                G_loss, recon_loss, G_loss_GAN = model.forward_G(audio, template, vertice, criterion,
                                                                 one_hot, D)
                valid_loss_log.append(recon_loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:,iter,:]
                    G_loss, recon_loss, G_loss_GAN = model.forward_G(audio, template, vertice, criterion,
                                                             one_hot, D)
                    valid_loss_log.append(recon_loss.item())
                        
        current_loss = np.mean(valid_loss_log)
        
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.7f}".format(e+1,current_loss))    
    return model

@torch.no_grad()
def test(args, model, test_loader,epoch):
    result_path = os.path.join(args.dataset,args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()
   
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze() # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze() # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save_GAN", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result_GAN", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument('--w_GAN', type=float, default=0.05, help='Weighting of GAN loss')
    parser.add_argument('--w_recon', type=float, default=1., help='Weighting of L1 loss')
    parser.add_argument('--GAN_type', type=str, default='TCN', help='Type of architecture for the discriminator')
    args = parser.parse_args()

    #build model
    model = FaceformerGAN(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()
    n_subjects = len(args.train_subjects.split(' '))
    D = Discriminator(args.vertice_dim + args.feature_dim, 512, model_type=args.GAN_type, n_cond=n_subjects).to(args.device)

    # Train the model
    G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr)
    model = trainer(args, dataset["train"], dataset["valid"], model, D, G_optimizer, D_optimizer, criterion, epoch=args.max_epoch)
    
    test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()