# Adversarial Training On General Voice Conversion <span style="color:red"></span>
Author: Hao-Chun Yang, Gao-Yi Chao, Hui-Ting Hong, Kun-Chieh Hsu </br>

## Introduction

<p align="center"><img src = "./imgs/intro.png"></p>

Voice conversion(VC) is a classical task which usually aims at transforming the voice of a source speaker into that of a target speaker while preserving the linguistic content[2]. In this project, we are going to deal with this classical task with unaligned corpora in conditional variational autoencoder with adversarial training.</br>
Many techniques have been developed for VC, such as Gaussian mixture models, Hidden Markov models, Gaussian regression, RNN. One big issue in the training of the speech conversion system is data, which prefers parallel and pairwise data. Since the inter-speaker speech alignment could be a problem for machines to learn speech style among different individuals. Hence, in this work, we apply adversarial training architecture for speech style transformation.
 

#### Files: </br>
* [vae-gan-hgy.py](./Scripts/vae-gan-hgy.py): for running the code.

## Methodology

<p align="center"><img src = "./imgs/metho.png"></p>

The structure we are going to use is the variational autoencoder combined with the discriminator for VC problem. First, we can decompose our autoencoder into two phases. During the first stage, the encoder encode a speaker independent latent representation z, which can be viewed as the phonetic-representation of the speeches. Then, in the second stage, z will be concatenated with speaker-identification-representation y, which in our experiments is a one-hot-encoding vector for distinguished subjects. The task of evaluating the similarity between reconstruct feature and target feature goes to discriminator. Here we use improved W-GAN, as the training objectives. Its discriminative power give us a supervision on judging whether the generated speech probability distribution is approximate to the real target speaker’s speech distribution. We believe that with the combination of conditional VAE  and adversarial training techniques, we are able to implement a voice conversion system with better acoustic style transfer result.</br>
Also, in our experiments, we applied both DNN and RNN version of the network. The main idea using RNN is that it can utilize their internal memory to process arbitrary sequences of inputs to exhibit the dynamic temporal behavior of a sequence, and we plan to use long short term memory networks (LSTM) as Encoder-Decoder structure to hopefully reconstruct indistinguishable feature compare with the target feature. Some results are given in the following section. 


## Results and Discussion
#### 1. Comparison between DNN and RNN.
Compare generated results using DNN and RNN framework. We found that the results of the RNN framework is quite unsatisfied. We guess the reason is that although recurrent neural network can preserve the memory information during training progress, however, the temporal information is not long enough (due to the hardware limitation, we only try on timestep < 200). This leads our autoencoder into quite under-training situation, resulting in terrible speech reconstruct ability. So in the following experiments, we all utilize the DNN as our final framework
#### 2. Training Variational Autoencoder.
During the training progress of the variational autoencoder, we found that the Kullback–Leibler divergence (KLD) oftenly converge so fast that it goes to zero within few iterations. This largely limit the reconstruction ability of our autoencoder because this restrict the variability of the latent representation z, leading to local optimum of the whole network. So here, we try different training scenarios and found that training merely reconstruct criterion a few iterations first, then following by joint training of the KLD and reconstruct error solved the problem.  

## Conclusion
Using a VAE-GAN structure can indeed solve the problem of the unaligned and inconsistent-content dataset and with several training tricks mentioned above,  we create a more robust model on multi-person voice conversion system. As the demo shown in the presentation we can hear the reconstructed voice is pretty nice, we believe with so many promising applications of voice conversion, including speech impaired, gamming, healthcare, etc. We can go way much longer in the future.

## Reference

[1] C. C. Hsu, et. al., “Voice conversion from unaligned corpora using variational autoencoding wasserstein generative adversarial networks”, arXiv preprint arXiv:1704.00849, 2017.</br>
[2]  T. Toda, L. Chen, D. Saito, F. Villavicencio, M. Wester, Z. Wu, and J. Yamagishi, “The Voice Conversion Challenge 2016,” in (submitted to) Interspeech, 2016.</br>
[3] L.-H. Chen, Z.-H. Ling, L.-J. Liu, and L.-R. Dai, “Voice conversion using deep neural networks with layer-wise generative training,” in Intelligent Control and Information Processing (ICICIP), vol. 22, no. 12, pp. 1859-1872, 2014. </br>
[4] L. Sun, S. Kang, K. Li, and H. Meng, “Voice conversion using deep Bidirectional Long Short-Term Memory based Recurrent Neural Networks,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015, pp. 4869-4873. </br>
[5] M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein GAN,” CoRR, vol. abs/1701.07875, 2017. [Online]. Available: http://arxiv.org/abs/1701.07875 
[6] Goodfellow, Ian; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Joshua (2014). "Generative Adversarial Networks". arXiv:1406.2661</br>
[7] http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/</br>
