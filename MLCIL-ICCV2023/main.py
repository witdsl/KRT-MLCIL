import datetime
import time
from MultiLabelIncremental_baseline import MultiLabelIncremental
import torch.distributed as dist
import argparse
import yaml
import torch
import os 

parser = argparse.ArgumentParser(description='MultilabelIncremental Training')
parser.add_argument('--local_rank', default=0, type=int, help='local rank for DistributedDataParallel')
parser.add_argument('--options', nargs='*')
parser.add_argument('--output_name', type=str)


def load_options(args, options):
    for o in options:
        with open(o) as f:
            config = yaml.safe_load(f)
            for k, v in config.items():
                setattr(args, k, v)


def main():
    args = parser.parse_args()
    if args.options:
        load_options(args, args.options)

    # Distributed
    if 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
    print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
          % (args.rank, args.world_size))

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=5400))

    args.logger_dir = 'logs/' + args.output_name 
    args.tensorboard_dir = 'tensorboard/' + args.output_name 
    args.model_save_path = 'saved_models/' + args.output_name 
    
    if args.rank == 0:
        print("Arguments:")
        for k, v in sorted(vars(args).items()):
            print(k, '=', v)

    if args.arch == 'baseline':
        multi_incremental = MultiLabelIncremental(args)
        multi_incremental.train()
    else:
        print('error')
    del multi_incremental

if __name__ == '__main__':
    main()
