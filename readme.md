# Fourier-Net

This is the official implementation of Fourier-Net: Fast Image Registration with Band-limited Deformation.

If you find the code helpful, please consider citing our work:

```
@article{jia2022fourier,
  title={Fourier-Net: Fast Image Registration with Band-limited Deformation},
  author={Jia, Xi and Bartlett, Joseph and Chen, Wei and Song, Siyang and Zhang, Tianyang and Cheng, Xinxing and Lu, Wenqi and Qiu, Zhaowen and Duan, Jinming},
  journal={arXiv preprint arXiv:2211.16342},
  year={2022}
}
```
## Updating
- [x] Update 2D Fourier-Net pre-trained models. Nov 29 2022.
- [x] Update 2D Fourier-Net training code. Mar 23 2023.
- [x] Update 2D Fourier-Net-Diff training code and pre-trained models. Mar 23 2023.
- [x] Update 3D Fourier-Net training code and pre-trained models. May 1 2023.
- [x] Update 3D Fourier-Net-Diff training code and pre-trained models. May 1 2023.

## Train and Test

```
# Train
CUDA_VISIBLE_DEVICES=0 python train.py --start_channel 8 --using_l2 2 --smth_labda 1.0 --lr 1e-4 --trainingset 4 --checkpoint 403 --iteration 403001
# Test
CUDA_VISIBLE_DEVICES=0 python infer.py --start_channel 8 --using_l2 2 --smth_labda 1.0 --lr 1e-4 --trainingset 4 --checkpoint 403 --iteration 403001
# Test with Bilinear Interplotation for Mask
CUDA_VISIBLE_DEVICES=0 python infer_bilinear.py --start_channel 8 --using_l2 2 --smth_labda 1.0 --lr 1e-4 --trainingset 4 --checkpoint 403 --iteration 403001
# Report Results
python compute_dsc_jet_from_quantiResult.py
```

## Acknowledgments

We would like to acknowledge the [IC-Net](https://github.com/zhangjun001/ICNet), [SYM-Net,](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) projects, from which we have adopted some of the code used in our work.

