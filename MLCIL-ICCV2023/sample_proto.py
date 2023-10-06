import time
import torch
import numpy as np


def _forward_features(model, loader):
    """
    Calculate features for icarl herding
    """
    features = []
    targets = []
    # forward features using model without ddp, only on 1 gpu
    model.eval()
    for i, (image, target) in enumerate(loader):
        image = image.cuda(non_blocking=True)
        with torch.no_grad():
            feature = model(image)['embeddings']
            features.append(feature)
            targets.append(target)
            
    features = torch.cat(features).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    return features, targets


def _icarl_selection(features, nb_examplars):
    features = np.array(features)
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
            np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]


def icarl_sample_protos(model, low_range, high_range, dataset, loader, num_protos, image_id_set=None, logger=None):
    """
    sample protos by icarl herding
    """
    if logger:
        logger.info('Herding, remove duplicate anchors after sampling')

    print('Save {} protos per class'.format(num_protos))

    # Create indices matrix
    all_indices = []
    for _ in range(low_range, high_range):
        class_indices = []
        all_indices.append(class_indices)

    start_time = time.time()
    features, targets = _forward_features(model, loader)
    indices = np.arange(len(features))
    if logger:
        logger.info('forward feature time:{:.2f}'.format(time.time()-start_time))
    
    # Assign features to regarding label
    start_time = time.time()
    for i, target in enumerate(targets):
        for label in target.nonzero()[0]:
            all_indices[label-low_range].append(indices[i])
    if logger:
        logger.info('Assign features to regarding label time:{:.2f}'.format(time.time()-start_time))

    # Select samples for each classes
    start_time = time.time()
    for i, label in enumerate(range(low_range, high_range)):
        class_idx = np.array(all_indices[i]) 
        # num_per_cls = int(num_protos*len(class_idx))
        num_per_cls = num_protos
        # print('Save {} protos for class {}'.format(num_per_cls, i))
        # if logger:
        #     logger.info('Save {} protos for class {}'.format(num_per_cls, i))

        if i == 0:
            all_herding_index = class_idx[_icarl_selection(features[class_idx], nb_examplars=num_per_cls)]
        else:
            cls_herding_index = class_idx[_icarl_selection(features[class_idx], nb_examplars=num_per_cls)]
            all_herding_index = np.concatenate((all_herding_index, cls_herding_index))
    if logger:
        logger.info('Select samples for each classes time:{:.2f}'.format(time.time()-start_time))
        
    # Remove duplicated
    if logger:
        logger.info('Before remove duplicated:{}'.format(len(all_herding_index)))

    # VOC hasn't ids attribute, can not remove duplicate anchors
    if hasattr(dataset, 'ids'):
        remove_dup_index = []
        for idx in all_herding_index:
            if dataset.ids[idx] in image_id_set:
                continue
            else:
                image_id_set.add(dataset.ids[idx])
                remove_dup_index.append(idx)
    else:
        remove_dup_index = all_herding_index

    if logger:
        logger.info('After remove duplicated:'.format(len(remove_dup_index)))

    herding_ds = torch.utils.data.Subset(dataset, remove_dup_index)
                
    return herding_ds


def random_sample_protos(dataset, low_range, high_range, num_protos):
    """
    Random sampling
    """
    num_classes = high_range - low_range
    random_idx = torch.randint(len(dataset), (num_classes * num_protos,))
    ramdom_sample_ds = torch.utils.data.Subset(dataset, random_idx)

    return ramdom_sample_ds