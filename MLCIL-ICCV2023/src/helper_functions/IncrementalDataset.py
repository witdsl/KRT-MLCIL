import os
import torch
import torch.distributed as dist
from .voc_loader import VOC
from .coco_loader import COCOLoader, coco_ids_to_cats, coco_fake2real


def build_dataset(dataset_name, root_dir, low_range, high_range, phase=None, year='2014', transform=None):
    """
    Get the training and val datasets of the given range
    """

    # dataset file name
    assert phase in ['train', 'val'], "phase should be \'train\' or \'val\'"
    dataset_name = dataset_name.lower()
    file_name = phase + year

    # Included Classes
    retrieve_classes = range(low_range, high_range)
    # Load dataset from home
    if 'coco' in dataset_name:
        instances_path = os.path.join(root_dir, f'annotations/instances_{file_name}.json')
        data_path = f'{root_dir}/{file_name}'

        dataset = COCOLoader(data_path, instances_path, included=retrieve_classes,
                             transform=transform)

    elif dataset_name == 'voc':
        if phase == 'train':
            dataset = VOC('07', 'edgeboxes', 'trainval', included=retrieve_classes, root=root_dir,
                          transform=transform)
        if phase == 'val':
            dataset = VOC('07', 'edgeboxes', 'test', included=retrieve_classes, root=root_dir,
                          transform=transform)

    return dataset


def build_loader(dataset, batch_size, num_workers, phase=None):
    assert phase in ['train', 'val'], "phase should be \'train\' or \'val\'"

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                                              shuffle=(phase == 'train'))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=num_workers, pin_memory=(phase == 'train'), sampler=sampler)

    return dataloader
