## Knowledge Restore and Transfer for Multi-Label Class-Incremental Learning
PyTorch code for the ICCV 2023 paper: <br />
Knowledge Restore and Transfer for Multi-Label Class-Incremental Learning  <br /> 
Songlin Dong, Haoyu Luo, Yuhang He, Xing Wei, Jie Cheng and Yihong Gong  <br />
IEEE/CVF International Conference on Computer Vision 2023   <br />
https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Knowledge_Restore_and_Transfer_for_Multi-Label_Class-Incremental_Learning_ICCV_2023_paper.html <br />
![PIC2_page-0001.jpg](https://s2.loli.net/2023/10/06/t34WSTOYE1seqlN.jpg)  
## Abstract
Current class-incremental learning research mainly focuses on single-label classification tasks while multi-label class-incremental learning (MLCIL) with more practical application scenarios is rarely studied. Although there have been many anti-forgetting methods to solve the problem of catastrophic forgetting in single-label class-incremental learning, these methods have difficulty in solving the MLCIL problem due to label absence and information dilution problems. To solve these problems, we propose a Knowledge Restore and Transfer (KRT) framework including a dynamic pseudo-label (DPL) module to solve the label absence problem by restoring the knowledge of old classes to the new data and an incremental cross-attention (ICA) module with session-specific knowledge retention tokens storing knowledge and a unified knowledge transfer token transferring knowledge to solve the information dilution problem. Comprehensive experimental results on MS-COCO and PASCAL VOC datasets demonstrate the effectiveness of our method for improving recognition performance and mitigating forgetting on multi-label class-incremental learning tasks.
## Task protocol
![PIC1-mlcil.jpg](https://s2.loli.net/2023/10/06/Q8J42HDepTa1sWw.jpg)
Data partitioning is found in src/helper_functions/IncrementalDataset.py
## Setup
Install anaconda: https://www.anaconda.com/distribution/ <br />
set up conda environment w/ python 3.7, ex: `conda create --name coda python=3.7` <br />
`conda activate coda`  <br />
`pip install -r requirements.txt`  <br />
NOTE: this framework was tested using `torch == 1.11.0` but should work for previous versions <br />
`timm ==0.5.4 inplace_abn=1.1.0` <br />
## Datasets and pretrained model
Download MS-COCO 2014 <br />
Put the dataset as ./datasets/coco2014 <br />
Donwload PASCAL VOC 2007 <br />
Put the dataset as ./datasets/voc2007 <br />
TResNetM pretrained on ImageNet 21k is available at [TResNetM_pretrained_model](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ASL/MS_COCO_TRresNet_M_224_81.8.pth), download it to ./pretrained_models and rename it as **tresnet_m_224_21k.pth**

## Training 
All commands should be run under the project root directory. The scripts are set up for 2 GPUs but can be modified for your hardware.
```bash
sh train_mlcil_coco.sh
sh train_mlcil_voc.sh
```
## Results
Results will be saved in a folder named `logs/`. To get the exprimental detail, retrieve the number in the file `logs/**/log/log.txt`
The model for the incremental stage is stored under `saved_models/`
## Citation
**If you found our work useful for your research, please cite our work**: <br />
 ```bash
        @InProceedings{dong2023knowledge,
          title={Knowledge Restore and Transfer for Multi-Label Class-Incremental Learning}, 
          author={Dong, Songlin and Luo, Haoyu and He, Yuhang and Wei, Xing and Cheng, Jie and Gong, Yihong},
          booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
          pages={18711--18720},
          year={2023}
        }
 ```
   
