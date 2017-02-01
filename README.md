# Wasserstein GAN - Implemented in TensorFlow + TFSlim

Wasserstein GAN (or WGAN) is a variant of Generative Adversarial Networks recently proposed by Martin Arjovsky, Soumith Chintala, and Léon Bottou, check out the paper [here](https://arxiv.org/abs/1701.07875) and the reference implementation [here](https://github.com/martinarjovsky/WassersteinGAN). Using some relatively simple algorithm changes (and less simple math), WGANs seem to greatly improve the notoriously unstable training of GANs for image generation.

### Warning
I have so far been unable to find a better way to implement weight clipping than by using a Python loop, which is pretty slow. I would very much appreciate any insight on how that step could be moved to the computation graph!


## Requirements
* Tensorflow
* Your own dataset loader

## How To

* Create some boilerplate code to load your data into batches and then place that where the "YOURDATAHERE" placeholder is (line 108)
* Fiddle with the parameters at the top of the file (and/or add your own architectures for the generator and discriminator)
* Run it!
` python train_WGAN.py `

The training can be stopped with Ctrl-C, which will cause it to save a checkpoint, so don't be surprised it doesn't shut down right away.

Due to the weight clipping step, the entire training uses the GPU rather inefficiently. Until I can find a better solution, you can increase the "clipping_per" parameter, so that the weights will only be clipped every so many steps, to gain some performance. This was not in the paper and I have no idea how it will affect the algorithm, so be warned!

You can easily drop in any replacement architectures for the discriminator or generator networks (just modify the generator and discriminator functions). WGAN is solely a training algorithm trick and should work with any architecture. The default nets are both fully convolution DCGANs for 128x128x3 images and NOT identical to those used in the paper.
