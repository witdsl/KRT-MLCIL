
****
#Paper 
PyTorch code for the ICCV 2023 paper: Knowledge Restore and Transfer for Multi-Label Class-Incremental Learning 

Songlin Dong, Haoyu Luo, Yuhang He, Xing Wei, Jie Cheng and Yihong Gong
https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Knowledge_Restore_and_Transfer_for_Multi-Label_Class-Incremental_Learning_ICCV_2023_paper.html

![PIC2_page-0001.jpg](https://s2.loli.net/2023/10/06/t34WSTOYE1seqlN.jpg)

****
# Preparation  
## Install required packages.  

    pip install -r requirments.txt
## Prepare Datasets  
Download MS-COCO 2014  
put the dataset at ./datasets/coco2014

Download PASCAL VOC 2007  
put the dastaset at ./datasets/voc2007

## Pretrained model  
TResNetM pretrained on ImageNet 21k is available at [TResNetM_pretrained_model](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ASL/MS_COCO_TRresNet_M_224_81.8.pth), download it to ./pretrained_models and rename it as **tresnet_m_224_21k.pth**

## Data partitioning
src/helper_functions/IncrementalDataset.py 

# Training

## Train on COCO

    # single node multi processes  
    python -m torch.distributed.launch --nproc_per_node 2 --master_port=21000 main.py \
    --options configs/multi_tresnetm_baseline_coco.yaml --output_name MICIL_COCO_POD

##  Train on VOC

    # single node multi processes  
    python -m torch.distributed.launch --nproc_per_node 2 --master_port=21000 main.py \
    --options configs/multi_tresnetm_baseline_voc.yaml --output_name MICIL_VOC_POD 
****
argments are provided by --options as a yaml file