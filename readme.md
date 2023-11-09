# Fourier-Net

This repository contains the official implementations of

* 1 **[Fourier-Net: Fast Image Registration with Band-limited Deformation](https://doi.org/10.1609/aaai.v37i1.25182) (AAAI-2023 oral presentation)**;
* 2 **[Fourier-Net+: Leveraging Band-Limited Representation for Efficient 3D Medical Image Registration](https://arxiv.org/abs/2307.02997)**.

## News

Oct 2023. Our Fourier-Net achieved overall 2nd place on [the recent Learn2Reg 2023 challenge](https://learn2reg.grand-challenge.org/Description/).

* 1 NLST: our submitted displacement is the algorithmic average of LKU-Net and Cascaded Fourier-Net+. **Best TRE_LM!**
* 2 ThoraxCBCT: our submitted displacement is solely from Fourier-Net. **Best TRE_LM!**

| team       | Task       | TRE LM | TRE LM30 | TRE nodules | Dice   | HD59    | SDLogJacDet |
|------------|------------|--------|----------|-------------|--------|---------|-------------|
| Birmingham | NLST       | 1.425  | 1.926    | 1.135       | N/A    | N/A     | 0.045       |
| Birmingham | ThoraxCBCT | 4.0342 | 3.8522   | N/A         | 0.5691 | 56.4998 | 0.0728      |

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

If you find the code helpful, please consider citing our work:

```
@inproceedings{jia2023fourier,
  title={Fourier-Net: Fast Image Registration with Band-Limited Deformation},
  author={Jia, Xi and Bartlett, Joseph and Chen, Wei and Song, Siyang and Zhang, Tianyang and Cheng, Xinxing and Lu, Wenqi and Qiu, Zhaowen and Duan, Jinming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={1},
  pages={1015--1023},
  year={2023}
}
```
```
@article{jia2023fourierplus,
  title={Fourier-Net+: Leveraging Band-Limited Representation for Efficient 3D Medical Image Registration},
  author={Jia, Xi and Thorley, Alexander and Gomez, Alberto and Lu, Wenqi and Kotecha, Dipak and Duan, Jinming},
  journal={arXiv preprint arXiv:2307.02997},
  year={2023}
}
```
## Updating
- [x] Update 2D Fourier-Net pre-trained models. Nov 29 2022.
- [x] Update 2D Fourier-Net training code. Mar 23 2023.
- [x] Update 2D Fourier-Net-Diff training code and pre-trained models. Mar 23 2023.
- [x] Update 3D Fourier-Net training code and pre-trained models. May 1 2023.
- [x] Update 3D Fourier-Net-Diff training code and pre-trained models. May 1 2023.
- [x] Update 2D Fourier-Net+ model. July 6 2023.
- [x] Update 3D Fourier-Net+ model. July 6 2023.
- [x] Update 2D Fourier-Net+ training/testing/pre-trained model. Nov 4 2023.
- [x] Update 3D Fourier-Net+ training/testing/pretrained model. Nov 4 2023.
## Acknowledgments

We would like to acknowledge the [IC-Net](https://github.com/zhangjun001/ICNet), [SYM-Net,](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) projects, from which we have adopted some of the code used in our work.

