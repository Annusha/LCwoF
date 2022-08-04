## Generalized and Incremental Few-Shot Learning by Explicit Learning and Calibration without Forgetting
Official Implementation, ICCV21

[arxiv](https://arxiv.org/abs/2108.08165)  
[Computer Vision Talks](https://www.youtube.com/watch?v=i6ZbnnKIACI)  
[poster](https://drive.google.com/file/d/1AaVD1x22c3wi0tNjmwtP1MLBpy8PnNin/view?usp=sharing)  |  [poster_slides](https://drive.google.com/file/d/18rVouHgWbUT5voy-vN4MdC_sRuMCzDTZ/view?usp=sharing) | [5_min_video](https://drive.google.com/file/d/1oFjWyuCM60XHfPbAKNLU7JwzBcSNHOVO/view?usp=sharing)


##### Data
[miniImageNet](https://drive.google.com/file/d/1CZPTOfQMp5ANF-BIuK9O9NdcPlT5XMHE/view?usp=sharing)  
here you can find all the splits and files for episodic training on mini-ImageNet (2.1 Gb)  

[pretrained_model](https://drive.google.com/file/d/165yPQtX1pWPZR_rBdPih2Rl1Xq6G7ln3/view?usp=sharing) on base 64 classes  
there is also script to train the model from scratch  
`python mini_imgnet/run_pretrain_base.py`

#### Run
set up all paths and run

`python mini_imagenet/run_novel.py
`


##### Logs

set up params for logging of training, metrics and visdom in `run_novel.py`


[files with harmonic mean and arithmetic mean](https://drive.google.com/file/d/1TIjWIOXzxPcHAfa1VTTk3p1OvcbvsNEK/view?usp=sharing) in different spaces (generalized and not), for 5w1s   
to save these files yourself turn `write_in_file=True`


###### details

pytorch 1.6  
[list of all packages that were installed](https://drive.google.com/file/d/178AdC8oQNJtJMeR78Ay4mdqhfJYWQyuY/view?usp=sharing) but you do not need all of them [just in case]

#### cite

@InProceedings{kukleva_lcwof,  
    author    = {Kukleva, Anna and Kuehne, Hilde and Schiele, Bernt},  
    title     = {Generalized and Incremental Few-Shot Learning by Explicit Learning and Calibration Without Forgetting},  
    booktitle = {ICCV},  
    year      = {2021},  
}
