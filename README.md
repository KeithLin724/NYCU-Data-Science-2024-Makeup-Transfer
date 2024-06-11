# NYCU Data Science 2024 - HW5 Makeup Transfer Using BeautyGAN

BeautyGAN: Instance-level Facial Makeup Transfer with Deep Generative Adversarial Network

---

## Requirements

First :

`pip install poetry`

Second :

`poetry install`

---

### Datasets

In this code we using `mtdataset`

### Training Code

`python train.py --data_path {your dataset path}`

For Tensorboard:

`tensorboard --logdir runs`, then open `http://localhost:6006/`

---

### TA readme

If you want to use this evaluation metric, you need prepare:

1. mtdataset/images/non-makeup
2. mtdataset/images/makeup
3. mt_removal
4. output_folder (the generated images)

The filenames of generated images should be as follows:
pred_0.png, pred_1.png, .....

The order of generated images are the makuep_test.txt and nomakup_test.txt .
For example, pred_0.png are generated by non-makeup/xfsy_0458.png and makeup/vHX44.png.
non-makeup/xfsy_0458.png and makeup/vHX44.png are the first line of nomakeup_test.txt and makup_test.txt, respectively

---

>### Acknowledgement
>
>This code is heavily based on [BeautyGAN - Offical Pytorch Implementation](https://github.com/wtjiang98/BeautyGAN_pytorch) and [BeautyGAN-PyTorch-reimplementation](https://github.com/thaoshibe/BeautyGAN-PyTorch-reimplementation). Thanks `wtjiang98`,`thaoshibe` so much to make his work available 🙏🙏🙏

> Package : [Poetry](./https://python-poetry.org/)
