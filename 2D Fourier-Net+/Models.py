import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Cascade(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(Cascade, self).__init__()
        self.net1 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.net2 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.net3 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.net4 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.warp = SpatialTransform()
        
    def forward(self, x, y):
        # in_pair = torch.cat((x, y), 1)
        
        
            
        X_temp = x.squeeze().squeeze()
        Y_temp = y.squeeze().squeeze()
        X_temp_fourier_all = torch.fft.fftn(X_temp)
        Y_temp_fourier_all = torch.fft.fftn(Y_temp)
        
        
        # X_temp_fourier_high = X_temp_fourier_all[40:120,48:144]#[40:120,48:144,84:140]
        X_temp_fourier_low = torch.fft.fftshift(X_temp_fourier_all)[40:120,48:144]#[40:120,48:144,84:140]
        # Y_temp_fourier_high = Y_temp_fourier_all[40:120,48:144]#[40:120,48:144,84:140]
        Y_temp_fourier_low = torch.fft.fftshift(Y_temp_fourier_all)[40:120,48:144]#[40:120,48:144,84:140]
        
        # X_temp_low_spatial_high = torch.real(torch.fft.ifftn(X_temp_fourier_high).unsqueeze(0).unsqueeze(0))
        X_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(X_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        # Y_temp_low_spatial_high = torch.real(torch.fft.ifftn(Y_temp_fourier_high).unsqueeze(0).unsqueeze(0))
        Y_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(Y_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
            
        
        out_1, out_2 = self.net1(X_temp_low_spatial_low, Y_temp_low_spatial_low)
        
        # out_1, out_2, out_3 = self.net(x, y)
        
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        # out_3 = out_3.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        # out_ifft3 = torch.fft.fftshift(torch.fft.fftn(out_3))
        # print(out_ifft1.shape)
        # p3d = (84, 84, 72, 72, 60, 60)
        p3d = (72, 72, 60, 60)
        out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
        # out_ifft3 = F.pad(out_ifft3, p3d, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))# * (img_x * img_y * img_z / 8))))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))# * (img_x * img_y * img_z / 8))))
        # disp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft3)))# * (img_x * img_y * img_z / 8))))
        # fxy_1 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0), disp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)
        fxy_1 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
            
        __, x2 = self.warp(x, fxy_1.permute(0, 2, 3, 1))
        
        
        X2_temp = x2.squeeze().squeeze()
        X2_temp_fourier_all = torch.fft.fftn(X2_temp)
        
        
        # X2_temp_fourier_high = X2_temp_fourier_all[40:120,48:144]#[40:120,48:144,84:140]
        X2_temp_fourier_low = torch.fft.fftshift(X2_temp_fourier_all)[40:120,48:144]#[40:120,48:144,84:140]
        # X2_temp_low_spatial_high = torch.real(torch.fft.ifftn(X2_temp_fourier_high).unsqueeze(0).unsqueeze(0))
        X2_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(X2_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        
        out_1, out_2 = self.net2(X2_temp_low_spatial_low, Y_temp_low_spatial_low)
        
        
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        # out_3 = out_3.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        # out_ifft3 = torch.fft.fftshift(torch.fft.fftn(out_3))
        # print(out_ifft1.shape)
        # p3d = (84, 84, 72, 72, 60, 60)
        p3d = (72, 72, 60, 60)
        out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
        # out_ifft3 = F.pad(out_ifft3, p3d, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))# * (img_x * img_y * img_z / 8))))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))# * (img_x * img_y * img_z / 8))))
        # disp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft3)))# * (img_x * img_y * img_z / 8))))
        # fxy_2 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0), disp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)
        fxy_2 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
            
        
        __, fxy_2_ = self.warp(fxy_1, fxy_2.permute(0, 2, 3, 1))
        
        fxy_2_ = fxy_2_ + fxy_2
        
        
        __, x3 = self.warp(x, fxy_2_.permute(0, 2, 3, 1))
        
        
        X3_temp = x3.squeeze().squeeze()
        X3_temp_fourier_all = torch.fft.fftn(X3_temp)
        
        
        # X2_temp_fourier_high = X2_temp_fourier_all[40:120,48:144]#[40:120,48:144,84:140]
        X3_temp_fourier_low = torch.fft.fftshift(X3_temp_fourier_all)[40:120,48:144]#[40:120,48:144,84:140]
        # X2_temp_low_spatial_high = torch.real(torch.fft.ifftn(X2_temp_fourier_high).unsqueeze(0).unsqueeze(0))
        X3_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(X3_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        
        out_1, out_2 = self.net3(X3_temp_low_spatial_low, Y_temp_low_spatial_low)
        
        
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        # out_3 = out_3.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        # out_ifft3 = torch.fft.fftshift(torch.fft.fftn(out_3))
        # print(out_ifft1.shape)
        p3d = (72, 72, 60, 60)
        out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
        # out_ifft3 = F.pad(out_ifft3, p3d, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))# * (img_x * img_y * img_z / 8))))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))# * (img_x * img_y * img_z / 8))))
        # disp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft3)))# * (img_x * img_y * img_z / 8))))
        # fxy_3 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0), disp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)
        fxy_3 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
            
        
        
        __, fxy_3_ = self.warp(fxy_2_, fxy_3.permute(0, 2, 3, 1))
        fxy_3_ = fxy_3_ + fxy_3
        
        
        
        __, x4 = self.warp(x, fxy_3_.permute(0, 2, 3, 1))
        
        
        X4_temp = x4.squeeze().squeeze()
        X4_temp_fourier_all = torch.fft.fftn(X4_temp)
        
        
        # X2_temp_fourier_high = X2_temp_fourier_all[40:120,48:144]#[40:120,48:144,84:140]
        X4_temp_fourier_low = torch.fft.fftshift(X4_temp_fourier_all)[40:120,48:144]#[40:120,48:144,84:140]
        # X2_temp_low_spatial_high = torch.real(torch.fft.ifftn(X2_temp_fourier_high).unsqueeze(0).unsqueeze(0))
        X4_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(X4_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        
        out_1, out_2 = self.net4(X4_temp_low_spatial_low, Y_temp_low_spatial_low)
        
        
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        # out_3 = out_3.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        # out_ifft3 = torch.fft.fftshift(torch.fft.fftn(out_3))
        # print(out_ifft1.shape)
        p3d = (72, 72, 60, 60)
        out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
        # out_ifft3 = F.pad(out_ifft3, p3d, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))# * (img_x * img_y * img_z / 8))))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))# * (img_x * img_y * img_z / 8))))
        # disp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft3)))# * (img_x * img_y * img_z / 8))))
        # fxy_3 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0), disp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)
        fxy_4 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
            
        
        
        __, fxy_4_ = self.warp(fxy_3_, fxy_4.permute(0, 2, 3, 1))
        fxy_4_ = fxy_4_ + fxy_4
        
        
        return fxy_4_

class UNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(UNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        f_xy = self.dc9(d3)
        #f_yx = self.dc10(d3)

        return f_xy#, f_yx


class Net_1_4(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(Net_1_4, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 4, bias=bias_opt)
        # self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        # self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 2, self.start_channel * 1, kernel_size=3,
                                stride=1, bias=bias_opt)
        # self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        # self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                # stride=1, bias=bias_opt)
        # self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        # self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                # stride=1, bias=bias_opt)
        # self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 1, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.up1 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up2 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        # self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        # self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
                # nn.Softsign())
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        # e4 = self.ec8(e3)
        # e4 = self.ec9(e4)

        # d0 = torch.cat((self.up1(e4), e3), 1)
        d0 = self.up1(e3)#torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        # d1 = torch.cat((self.up2(d0), e2), 1)
        d1 = self.up2(d0)#torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        # d1 = self.dc4(d1)

        # d2 = torch.cat((self.up3(d1), e1), 1)

        # d2 = self.dc5(d2)
        # d2 = self.dc6(d2)

        # d3 = torch.cat((self.up4(d2), e0), 1)
        # d3 = self.dc7(d3)
        # d3 = self.dc8(d3)

        f_xy = self.dc9(d1)
        #f_yx = self.dc10(d3)

        return f_xy[:,0:1,:,:], f_xy[:,1:2,:,:]#, f_yx

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        # d2, h2, w2 = mov_image.shape[-3:]
        h2, w2 = mov_image.shape[-2:]
        # grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        # grid_d = grid_d.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        # grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_h = flow[:,:,:,0]
        flow_w = flow[:,:,:,1]
        # flow_w = flow[:,:,:,2]
        #Softsign
        #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
        #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
        #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
        
        # Remove Channel Dimension
        # disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True,padding_mode="border")
        
        return sample_grid, warped

class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
        h2, w2, = flow.shape[-2:]
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        
        
        for i in range(self.time_step):
            flow_h = flow[:,0,:,:]
            flow_w = flow[:,1,:,:]
            # print(flow.shape)
            # print(flow_h.shape)
            # print(grid_h.shape)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)
            
            deformation = torch.stack((disp_w,disp_h), dim=3)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear',padding_mode="border", align_corners=True)
            #Softsign
            #disp_d = (grid_d + (flow_d * 2 / d2)).squeeze(1)
            #disp_h = (grid_h + (flow_h * 2 / h2)).squeeze(1)
            #disp_w = (grid_w + (flow_w * 2 / w2)).squeeze(1)
            
            # Remove Channel Dimension
            #disp_h = (grid_h + (flow_h)).squeeze(1)
            #disp_w = (grid_w + (flow_w)).squeeze(1)

            #sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, H, W, 2)
            #flow = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod)
        
        return flow


class CompositionTransform(nn.Module):
    def __init__(self):
        super(CompositionTransform, self).__init__()

    def forward(self, flow_1, flow_2, sample_grid, range_flow):
        size_tensor = sample_grid.size()
        grid = sample_grid + (flow_1.permute(0,2,3,4,1) * range_flow)
        grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
        grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
        grid[:, :, :, :, 2] = (grid[:, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
        compos_flow = F.grid_sample(flow_2, grid, mode='bilinear', align_corners = True,padding_mode="border") + flow_1
        return compos_flow


def smoothloss(y_pred):
    #print('smoothloss y_pred.shape    ',y_pred.shape)
    #[N,3,D,H,W]
    h2, w2 = y_pred.shape[-2:]
    # dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dz*dz))/2.0


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


def magnitude_loss(flow_1, flow_2):
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag))/num_ele

    return diff

"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv2d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)

class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice



class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
class SAD:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))
