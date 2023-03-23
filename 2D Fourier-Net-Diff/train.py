import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from Models import *
from Functions import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
import csv
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from natsort import natsorted

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=1000.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--magnitude", type=float,
                    dest="magnitude", default=0.001,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--mask_labda", type=float,
                    dest="mask_labda", default=0.25,
                    help="mask_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.02,
                    help="labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=800,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    #default='/export/local/xxj946/AOSBraiCN2',
                    default='/bask/projects/d/duanj-ai-imaging/Accreg/brain/OASIS_AffineData/',
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=3,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
local_ori = opt.local_ori
magnitude = opt.magnitude
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
datapath = opt.datapath
mask_labda = opt.mask_labda
data_labda = opt.data_labda
trainingset = opt.trainingset
using_l2 = opt.using_l2

def dice(pred1, truth1):
    mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    dice_list=[]
    for k in mask_value4[1:]:
        #print(k)
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        # print(intersection)
        dice_list.append(intersection / (np.sum(pred) + np.sum(truth)))
    return np.mean(dice_list)

def save_checkpoint(state, save_dir, save_filename, max_model_num=6):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
def train():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SYMNet(2, 2, start_channel).cuda()
    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC(win=9)
    elif using_l2 == 3:
        ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1, win_size=9)
        ssim_module = SSIM(data_range=1, size_average=True, channel=1) # channel=1 for grayscale images
        loss_similarity = SAD().loss
    loss_smooth = smoothloss
#    loss_magnitude = magnitude_loss
#    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform().cuda()
    # init_dict = torch.load('/bask/projects/d/duanj-ai-imaging/Accreg/brain_half/Learn_FFT_404856/L2ss_1_Chan_8_Smth_1000000.0_LR_0.0001/SYMNet_320000.pth')
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
#    com_transform = CompositionTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    # for name, param in model.named_parameters():
        # param.requires_grad = True
        # with torch.no_grad():
            # if name in init_dict.keys():
                # print(name)
                # param.copy_(init_dict[name])

                                                                                                                                         
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    lossall = np.zeros((3, iteration))
    train_set = TrainDataset(datapath,img_file='train_list.txt',trainingset = trainingset)
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    test_set = ValidationDataset(opt.datapath,img_file='val_list.txt')
    test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)
    model_dir = './L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_Pth/'.format(using_l2,start_channel,smooth, trainingset, lr)
    model_dir_png = './L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_Png/'.format(using_l2,start_channel,smooth, trainingset, lr)
    csv_name = 'L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}.csv'.format(using_l2,start_channel,smooth, trainingset, lr)
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    if not os.path.isdir(model_dir_png):
        os.mkdir(model_dir_png)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # model_dir_samples = './L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_Samples'.format(using_l2,start_channel,smooth, trainingset, lr)

    # if not os.path.isdir(model_dir_samples):
        # os.mkdir(model_dir_samples)
    
    
    step = 1

    while step <= iteration:
        for X, Y in training_generator:

            X = X.cuda().float()

            Y = Y.cuda().float()
            
            out_1, out_2 = model(X, Y)
            out_1 = out_1.squeeze().squeeze()
            out_2 = out_2.squeeze().squeeze()
            out_ifft1 = torch.fft.fftshift(torch.fft.fft2(out_1))
            out_ifft2 = torch.fft.fftshift(torch.fft.fft2(out_2))
            # p3d = (84, 84, 70, 70)
            p3d = (72, 72, 60, 60)
            out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
            out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
            # out_ifft3 = F.pad(out_ifft3, p3d, "constant", 0)
            disp_mf_1 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft1)))# * (img_x * img_y * img_z / 8))))
            disp_mf_2 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft2)))# * (img_x * img_y * img_z / 8))))
            # disp_mf_3 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft3)))# * (img_x * img_y * img_z / 8))))
            f_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
            
            D_f_xy = diff_transform(f_xy)
            grid, X_Y = transform(X, D_f_xy.permute(0, 2, 3, 1))
            
                                                                     
            if using_l2 == 3:
                loss1 = (1 - ms_ssim_module(X_Y , Y)) + (1 - ssim_module(X_Y , Y))# + 0 * loss_similarity(Y, X_Y)
            else:
                loss1 = loss_similarity(Y, X_Y)
            loss5 = loss_smooth(f_xy)
            
            loss = loss1 + smooth * loss5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item(),loss1.item(),loss5.item()])
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" -smo "{3:.4f}" '.format(step, loss.item(),loss1.item(),loss5.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):
                with torch.no_grad():
                    Dices_Validation = []
                    # Dices_Validation_Sup = []
                    for __, __, mov_img, fix_img, mov_lab, fix_lab in test_generator:
                        model.eval()
                        
                        # sup_v_x, sup_v_y  = supvised_model(mov_img.float().to(device), fix_img.float().to(device))
                        # sup_v_x_fft = torch.fft.fftshift(torch.fft.fft2(sup_v_x.squeeze().squeeze()))
                        # sup_v_y_fft = torch.fft.fftshift(torch.fft.fft2(sup_v_y.squeeze().squeeze()))
                        # sup_v_x = torch.real(torch.fft.ifft2(torch.fft.ifftshift(sup_v_x_fft * mask)))# * (img_x * img_y * img_z / 8))))
                        # sup_v_y = torch.real(torch.fft.ifft2(torch.fft.ifftshift(sup_v_y_fft * mask)))# * (img_x * img_y * img_z / 8))))
                        # print(sup_v_y.shape)
                        # sup_v_xy = torch.cat([sup_v_x.unsqueeze(0).unsqueeze(0), sup_v_y.unsqueeze(0).unsqueeze(0)], dim = 1)
                        
                        vout_1, vout_2 = model(mov_img.float().to(device), fix_img.float().to(device))
                        vout_1 = vout_1.squeeze().squeeze()
                        vout_2 = vout_2.squeeze().squeeze()
                        vout_ifft1 = torch.fft.fftshift(torch.fft.fft2(vout_1))
                        vout_ifft2 = torch.fft.fftshift(torch.fft.fft2(vout_2))
                        # p3d = (72, 72, 60, 60)
                        # p3d = (84, 84, 70, 70)
                        p3d = (72, 72, 60, 60)
                        # p3d = (90, 90, 75, 75)
                        # p3d = (84, 84, 70, 70)
                        # p3d = (84, 84, 70, 70)
                        vout_ifft1 = F.pad(vout_ifft1, p3d, "constant", 0)
                        vout_ifft2 = F.pad(vout_ifft2, p3d, "constant", 0)
                        vdisp_mf_1 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(vout_ifft1)))# * (img_x * img_y * img_z / 8))))
                        vdisp_mf_2 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(vout_ifft2)))# * (img_x * img_y * img_z / 8))))
                        vf_xy = torch.cat([vdisp_mf_1.unsqueeze(0).unsqueeze(0), vdisp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
                        D_vf_xy = diff_transform(vf_xy)
    #            print(F_xy.max())
                        __, warped_xv_seg= transform(mov_lab.float().to(device), D_vf_xy.permute(0, 2, 3, 1), mod = 'nearest')
                        # __, warped_xv_seg_sup = transform(mov_lab.float().to(device), sup_v_xy.permute(0, 2, 3, 1), mod = 'nearest')
                        for bs_index in range(bs):
                            dice_bs=dice(warped_xv_seg[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy())
                            Dices_Validation.append(dice_bs)
                            # dice_bs=dice(warped_xv_seg_sup[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy())
                            # Dices_Validation_Sup.append(dice_bs)
                    modelname = 'DiceVal_{:.5f}_Epoch_{:09d}.pth'.format(np.mean(Dices_Validation),step)
                    csv_dice = np.mean(Dices_Validation)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice])
                    save_checkpoint(model.state_dict(), model_dir, modelname)
                    np.save(model_dir_png + 'Loss.npy', lossall)
            if (step % n_checkpoint == 0):
                sample_path = os.path.join(model_dir_png, '{:09d}-images.jpg'.format(step))
                # h, w = f_xy.shape[-2:]
                # grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])
                # s = torch.stack((grid_w,grid_h), dim=0).cuda().float()
                # print(s.unsqueeze(0).shape)
                # print(mov_img.shape)
                # s = transform(s.unsqueeze(0),f_xy.permute(0, 2, 3, 1))
                save_flow(X, Y, X_Y, grid.permute(0, 3, 1, 2), sample_path)
                    # print('Saved debug images into {}...'.format(sample_path))
                     

            if step > iteration:
                break
            step += 1
        print("one epoch pass")
        # epoch = epoch + 1
    np.save(model_dir_png + '/loss_SYMNet.npy', lossall)
def save_flow(X, Y, X_Y, f_xy, sample_path):
    x = X.data.cpu().numpy()
    y = Y.data.cpu().numpy()
#    print('AAAAAAAAAAAAAAAAAAAAAAAA shape: {}, {}, {}, {}'.format(x.shape, pred.shape, x_pred.shape, flow.shape))
    # pred = pred.data.cpu().numpy()
    x_pred = X_Y.data.cpu().numpy()
    # pred = pred[0,...]
    x_pred = x_pred[0,...]
    x = x[0,...]
    y = y[0,...]
    
    flow = f_xy.data.cpu().numpy()
    op_flow =flow[0,:,:,:]
    # quiver_flow = op_flow.copy()
    # op_flow[0, :, :] = op_flow[0, :, :] / 2 * op_flow.shape[-2]
    # op_flow[1, :, :] = op_flow[1, :, :] / 2 * op_flow.shape[-1]


#    print(pred.max())
    plt.subplots(figsize=(7, 4))
    # plt.subplots()
    plt.subplot(231)
    plt.imshow(x[0, :, :], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(y[0, :, :], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(x_pred[0, :, :], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(234)
    # plt.subplot(245)
    # plt.imshow(x_pred[0, :, :], cmap='gray')
    # plt.axis('off')
    # plt.subplot(246)
    # plt.imshow(x[0, :, :], cmap='gray')
    interval = 5
    for i in range(0,op_flow.shape[1]-1,interval):
        plt.plot(op_flow[0,i,:], op_flow[1,i,:],c='g',lw=1)
    #plot the vertical lines
    for i in range(0,op_flow.shape[2]-1,interval):
        plt.plot(op_flow[0,:,i], op_flow[1,:,i],c='g',lw=1)
#    plt.axis((-1,1,-1,1))
    #plt.axis('equal')  
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(abs(x[0, :, :]-y[0, :, :]), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(abs(x_pred[0, :, :]-y[0, :, :]), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(sample_path,bbox_inches='tight')
    plt.close()
train()
