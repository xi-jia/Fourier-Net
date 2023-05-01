import glob
import os, utils
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
# from models.TransMorph import CONFIGS as CONFIGS_TM
# import models.TransMorph  as TransMorph
from Models import *

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
                    dest="magnitude", default=1000.0,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.25,
                    help="smth_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--fft_labda", type=float,
                    dest="fft_labda", default=0.02,
                    help="fft_labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=403,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
# parser.add_argument("--datapath", type=str,
                    # dest="datapath",
                    #default='/export/local/xxj946/AOSBraiCN2',
                    # default='/bask/projects/d/duanj-ai-imaging/Accreg/brain/OASIS_AffineData/',
                    # help="data path for training images")
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
# smooth = opt.smth_labda
# datapath = opt.datapath
smooth = opt.smth_labda
data_labda = opt.data_labda
# fft_labda = opt.fft_labda
# trainingset = opt.trainingset
using_l2 = opt.using_l2

def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = SpatialTransform().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    atlas_dir = '/bask/projects/d/duanj-ai-imaging/TransMorph_Xi/IXI_Mine/IXI_data/atlas.pkl'
    test_dir = '/bask/projects/d/duanj-ai-imaging/TransMorph_Xi/IXI_Mine/IXI_data/Test/'
    model_idx = -2
    model_dir = './L2ss_{}_Chan_{}_Smth_{}_LR_{}/'.format(using_l2, start_channel, smooth, lr)
    dict = utils.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+model_dir[:-1]+'_Test.csv'):
        os.remove('Quantitative_Results/'+model_dir[:-1]+'_Test.csv')
    csv_writter(model_dir[:-1], 'Quantitative_Results/' + model_dir[:-1]+'_Test')
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + model_dir[:-1]+'_Test')

    
    model = SYMNet(2, 3, start_channel).cuda()
    
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])#['state_dict']
    model.load_state_dict(best_model, strict=False)
    model.cuda()
    # reg_model = utils.register_model(config.img_size, 'nearest')
    # reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            vout_1, vout_2, vout_3  = model(x.float().to(device), y.float().to(device))
            vout_1 = vout_1.squeeze().squeeze()
            vout_2 = vout_2.squeeze().squeeze()
            vout_3 = vout_3.squeeze().squeeze()
            vout_ifft1 = torch.fft.fftshift(torch.fft.fftn(vout_1))
            vout_ifft2 = torch.fft.fftshift(torch.fft.fftn(vout_2))
            vout_ifft3 = torch.fft.fftshift(torch.fft.fftn(vout_3))
            p3d = (84, 84, 72, 72, 60, 60)
            vout_ifft1 = F.pad(vout_ifft1, p3d, "constant", 0)
            vout_ifft2 = F.pad(vout_ifft2, p3d, "constant", 0)
            vout_ifft3 = F.pad(vout_ifft3, p3d, "constant", 0)
            vdisp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft1)))# * (img_x * img_y * img_z / 8))))
            vdisp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft2)))# * (img_x * img_y * img_z / 8))))
            vdisp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft3)))# * (img_x * img_y * img_z / 8))))
            vf_xy = torch.cat([vdisp_mf_1.unsqueeze(0).unsqueeze(0), vdisp_mf_2.unsqueeze(0).unsqueeze(0), vdisp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)
            D_f_xy = diff_transform(vf_xy)
            # def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            def_out= transform(x_seg.float().to(device), D_f_xy.permute(0, 2, 3, 4, 1), mod = 'nearest')
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            # print(f_xy.shape) #[1, 3, 160, 192, 224]
            dd, hh, ww = D_f_xy.shape[-3:]
            D_f_xy = D_f_xy.detach().cpu().numpy()
            D_f_xy[:,0,:,:,:] = D_f_xy[:,0,:,:,:] * dd / 2
            D_f_xy[:,1,:,:,:] = D_f_xy[:,1,:,:,:] * hh / 2
            D_f_xy[:,2,:,:,:] = D_f_xy[:,2,:,:,:] * ww / 2
            # jac_det = utils.jacobian_determinant_vxm(f_xy.detach().cpu().numpy()[0, :, :, :, :])
            jac_det = utils.jacobian_determinant_vxm(D_f_xy[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line +','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + model_dir[:-1]+'_Test')
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    # GPU_iden = 1
    # GPU_num = torch.cuda.device_count()
    # print('Number of GPU: ' + str(GPU_num))
    # for GPU_idx in range(GPU_num):
        # GPU_name = torch.cuda.get_device_name(GPU_idx)
        # print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    # torch.cuda.set_device(GPU_iden)
    # GPU_avai = torch.cuda.is_available()
    # print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    # print('If the GPU is available? ' + str(GPU_avai))
    main()