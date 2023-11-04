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
from natsort import natsorted
import csv

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=10, help="batch_size")
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
                    dest="checkpoint", default=4000,
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
                    dest="trainingset", default=4,
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
    dice_k=[]
    # mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    # print(len(mask4_value2))
    # mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    for k in mask4_value2[1:]:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        dice_k.append(intersection / (np.sum(pred) + np.sum(truth)))
    return np.mean(dice_k)

def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def train():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Cascade(2, 2, start_channel).cuda()
    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC(win=9)
    elif using_l2 == 3:
        ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1, win_size=9)
        loss_similarity = SAD().loss
    loss_smooth = smoothloss
#    loss_magnitude = magnitude_loss
#    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
#    com_transform = CompositionTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)
    #aos_params = list(model.ic_block.param)
    #other_params = [p for p in model.parameters() if p not in aos_params]
    #aos_params = [p for n,p in model.named_parameters() if n.startswith('ic_block.')]
    #other_params = [p for n,p in model.named_parameters() if not n.startswith('ic_block.')]
    #optimizer = torch.optim.Adam([{'params': other_params},{'params': aos_params, 'lr': 1e-4}], lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    lossall = np.zeros((3, iteration))
    train_set = TrainDataset(datapath,img_file='train_list.txt',trainingset = trainingset)
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    test_set = ValidationDataset(opt.datapath,img_file='val_list.txt')
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=2)
    model_dir = './L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}/'.format(using_l2, start_channel, smooth, trainingset, lr, bs)
    model_dir_pth1 = './L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Pth1/'.format(using_l2, start_channel, smooth, trainingset, lr, bs)
    model_dir_pth2 = './L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Pth2/'.format(using_l2, start_channel, smooth, trainingset, lr, bs)
    csv_name = 'L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}.csv'.format(using_l2, start_channel, smooth, trainingset, lr, bs)
    if os.path.exists(csv_name):
        assert 0==1
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(model_dir_pth1):
        os.mkdir(model_dir_pth1)
    if not os.path.isdir(model_dir_pth2):
        os.mkdir(model_dir_pth2)
    
    
    step = 1

    while step <= iteration:
        for mov_img, fix_img in training_generator:

            fix_img = fix_img.cuda().float()

            mov_img = mov_img.cuda().float()

            # fix_lab = fix_lab.cuda().float()

            # mov_lab = mov_lab.cuda().float()
            
            f_xy = model(mov_img, fix_img)
            # Df_xy = diff_transform(f_xy)
            Df_xy = f_xy
            __, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
           
            loss1 = loss_similarity(fix_img, warped_mov) # GT shall be 1st Param
            loss5 = loss_smooth(f_xy)
            
            loss = loss1 + smooth * loss5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item(),loss1.item(),loss5.item()])
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim "{2:.4f}" -smo "{3:.4f}" '.format(step, loss.item(),loss1.item(),loss5.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0) or (step==1):
                with torch.no_grad():
                    Dices_Validation = []
                    for __, __, vmov_img, vfix_img, vmov_lab, vfix_lab in test_generator:
                        model.eval()
                        V_xy = model(vmov_img.float().to(device), vfix_img.float().to(device))
                        # DV_xy = diff_transform(V_xy)
                        DV_xy = V_xy
                        # x_seg_oh = nn.functional.one_hot(mov_lab.long(), num_classes=25)
                        # x_seg_oh = torch.squeeze(x_seg_oh, 1)
                        # x_seg_oh = x_seg_oh.permute(0, 3, 1, 2).contiguous()
                        # grid, x_segs = transform(x_seg_oh.float().to(device), DV_xy.permute(0, 2, 3, 1), mod = 'bilinear')
                        #x_segs = model.spatial_trans(x_seg.float(), flow.float())
                        # x_segs = []
                        # for i in range(25):
                            # __, def_seg = transform(x_seg_oh[:, i:i + 1, ...].float().to(device), DV_xy.permute(0, 2, 3, 1))
                            # x_segs.append(def_seg)
                        # x_segs = torch.cat(x_segs, dim=1)
                        # warped_mov_lab = torch.argmax(x_segs, dim=1, keepdim=True)
                        grid, warped_vmov_lab = transform(vmov_lab.float().to(device), DV_xy.permute(0, 2, 3, 1), mod = 'nearest')
                        # for bs_index in range(1):
                        dice_bs = dice(warped_vmov_lab[0,...].data.cpu().numpy().copy(),vfix_lab[0,...].data.cpu().numpy().copy())
                        Dices_Validation.append(dice_bs)
                    modelname = 'DiceVal_{:.4f}_Step_{:09d}.pth'.format(np.mean(Dices_Validation), step)
                    csv_dice = np.mean(Dices_Validation)
                    if step <= iteration / 2.0:
                        save_checkpoint(model.state_dict(), model_dir_pth1, modelname)
                    else:
                        save_checkpoint(model.state_dict(), model_dir_pth2, modelname)
                    np.save(model_dir + 'Loss.npy', lossall)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice])
            if (step % n_checkpoint == 0):
                sample_path = os.path.join(model_dir, '{:08d}-images.jpg'.format(step))
                # h, w = f_xy.shape[-2:]
                # grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)])
                # s = torch.stack((grid_w,grid_h), dim=0).cuda().float()
                # print(s.unsqueeze(0).shape)
                # print(mov_img.shape)
                # s = transform(s.unsqueeze(0),f_xy.permute(0, 2, 3, 1))
                save_flow(mov_img, fix_img, warped_mov, grid.permute(0, 3, 1, 2), sample_path)
            step += 1

            if step > iteration:
                break
        print("one epoch pass")
    np.save(model_dir + '/Loss.npy', lossall)

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
    interval = 7
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
