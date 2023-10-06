


 voc dataset
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port=21000 main.py \
--options configs/multi_tresnetm_baseline_voc.yaml --output_name M21K_224_B4e5I1e4_ep20_2gpu_bs64_baseline_herding20