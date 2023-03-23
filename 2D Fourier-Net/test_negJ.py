import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import ValidationDataset
import torch.utils.data as Data
import csv

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


def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    # disp = disp.transpose(1, 2, 3, 0)
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    
    import pystrum.pynd.ndutils as nd
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


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

def test(modelpath):
    bs = 1
    model = SYMNet(2, 2, opt.start_channel).cuda()
    torch.backends.cudnn.benchmark = True
    transform = SpatialTransform().cuda()
    model.load_state_dict(torch.load(modelpath))
    #model_lambda = model.ic_block.labda.data.cpu().numpy()
    #model_odr = model.ic_block.odr.data.cpu().numpy()
    model.eval()
    transform.eval()
#    diff_transform.eval()
#    com_transform.eval()
#    Dices_before=[]
    Dices_35=[]
    NegJ_35=[]
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    test_set = ValidationDataset(opt.datapath,img_file='test_list.txt')
    test_generator = Data.DataLoader(dataset=test_set, batch_size=bs,
                                         shuffle=False, num_workers=2)
    for __, __, mov_img, fix_img, mov_lab, fix_lab in test_generator:
        with torch.no_grad():
            out_1, out_2 = model(mov_img.float().to(device), fix_img.float().to(device))
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
            V_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
            
            __, warped_mov_lab = transform(mov_lab.float().to(device), V_xy.permute(0, 2, 3, 1), mod = 'nearest')
            hh, ww = V_xy.shape[-2:]
            V_xy = V_xy.detach().cpu().numpy()
            V_xy[:,0,:,:] = V_xy[:,0,:,:] * hh / 2
            V_xy[:,1,:,:] = V_xy[:,1,:,:] * ww / 2
            #print('V_xy.shape . . . ', V_xy.shape)  #([1, 3, 160, 192, 224])
            #print('warped_mov_lab.shape . . . ', warped_mov_lab.shape) #([1, 1, 160, 192, 224])
            
            for bs_index in range(bs):
                dice_bs = dice(warped_mov_lab[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy())
                Dices_35.append(dice_bs)
                jac_det = jacobian_determinant_vxm(V_xy[0, :, :, :])
                negJ = np.sum(jac_det <= 0) / 160 / 192 * 100
                NegJ_35.append(negJ)

    print(np.mean(Dices_35), np.std(Dices_35))
    print('%100:', np.mean(NegJ_35), '%100:',np.std(NegJ_35))

    # return np.mean(Dices_35)


if __name__ == '__main__':
    # DICESCORES4=[]
    # DICESCORES35=[]
    model_path='./L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_Pth/'.format(opt.using_l2, opt.start_channel, opt.smth_labda, opt.trainingset, opt.lr)
    model_idx = -1
    from natsort import natsorted
    print('Best model: {}'.format(natsorted(os.listdir(model_path))[model_idx]))
    # best_model = torch.load(model_path + natsorted(os.listdir(model_path))[model_idx])#['state_dict']
    test(model_path + natsorted(os.listdir(model_path))[model_idx])
    
    
    # from torchsummary import summary
    # from Models import *

    # model = SYMNet(2, 2, opt.start_channel)
    # summary(model, [(1, 160, 192),(1, 160, 192)])
    # summary(model, (1, 128,128))#,(1, 80, 96, 112)])
    
    # csvname = 'L2ss_{}_Chan_{}_Smth_{}_Set_{}_Test.csv'.format(opt.using_l2, opt.start_channel, opt.smth_labda, opt.trainingset)
    # f = open(csvname, 'w')

    # with f:
        # fnames = ['Index','Dice35']
        # writer = csv.DictWriter(f, fieldnames=fnames)
        # writer.writeheader()
    # try:
        # for i in range(opt.checkpoint,opt.iteration,opt.checkpoint):
            # model_path='./L2ss_{}_Chan_{}_Smth_{}_Set_{}/SYMNet_{}.pth'.format(opt.using_l2, opt.start_channel, opt.smth_labda, opt.trainingset, i)
            # print(model_path)
            # dice35_temp= test(model_path)
            # f = open(csvname, 'a')
            # with f:
                # writer = csv.writer(f)
                # writer.writerow([i,dice35_temp])
            # DICESCORES35.append(dice35_temp)
    # except:
        # print(np.argmax(DICESCORES35))
        # print(np.max(DICESCORES35))
    # print(np.argmax(DICESCORES35))
    # print(np.max(DICESCORES35))