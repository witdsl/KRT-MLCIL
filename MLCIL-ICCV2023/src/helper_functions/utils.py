import sys
from torch.utils.tensorboard import SummaryWriter
from src.helper_functions.logger import setup_logger
import os 
import time
import pandas as pd


def build_logger(logger_dir, rank):
    logger = setup_logger(output=logger_dir + '/log',
                          distributed_rank=rank, color=False, name="MultiLabelIncremental")
    logger.info("Command: " + ' '.join(sys.argv))
    return logger


def build_writer(tensorboard_dir, args):
    writer = SummaryWriter(tensorboard_dir)
    writer.add_text('Args', str(vars(args)))

    return writer


def print_to_excel(excel_path, expe_name, dataset_name, base_classes, task_size, total_classes, params, map, metrics, git_hash):
    # read excel content
    sheet_name=f"{dataset_name} {base_classes}+{task_size}"

    # Results
    incremental_stages = [(0, base_classes)] + [
                (low, low + task_size) for low in range(base_classes, total_classes, task_size)]
    columns = ['date', 'name', 'params']
    for low_range, high_range in incremental_stages:
        columns.append(f"{low_range}-{high_range}")
    columns.append('avg_mAP')
    for metric in ["precision_c", "recall_c", "f1_c", "precision_o", "recall_o", "f1_o"]:
        columns.append(metric)
    columns.append('git hash')

    Current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    avg_map = sum(map)/len(map)
    new_result = pd.DataFrame([[Current_time, expe_name, params, *map, avg_map, *metrics, git_hash]], columns=columns)
    
    result = new_result
    if os.path.exists(excel_path):
        old_results = pd.read_excel(excel_path, sheet_name=None)
        if sheet_name in old_results.keys():
            old_results = old_results[sheet_name]
            result = old_results.append(new_result)
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            result.to_excel(writer, f'{dataset_name} {base_classes}+{task_size}', index=False)
    
    else:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            result.to_excel(writer, f'{dataset_name} {base_classes}+{task_size}', index=False)


def calculate_metrics(preds, targets, thre):

    prediction = preds.gt(thre).long()
    tp_c = (prediction + targets).eq(2).sum(dim=0)
    fp_c = (prediction - targets).eq(1).sum(dim=0)
    fn_c = (prediction - targets).eq(-1).sum(dim=0)
    tn_c = (prediction + targets).eq(0).sum(dim=0)
    count = targets.size(0)

    precision_c = [float(tp_c[i].float() / (tp_c[i] + fp_c[i]).float()) * 100.0 if tp_c[i] > 0 else 0.0 for i in range(len(tp_c))]
    recall_c = [float(tp_c[i].float() / (tp_c[i] + fn_c[i]).float()) * 100.0 if tp_c[i] > 0 else 0.0 for i in range(len(tp_c))]
    f1_c = [2 * precision_c[i] * recall_c[i] / (precision_c[i] + recall_c[i]) if tp_c[i] > 0 else 0.0 for i in range(len(tp_c))]

    mean_p_c = sum(precision_c) / len(precision_c)
    mean_r_c = sum(recall_c) / len(recall_c)
    mean_f_c = sum(f1_c) / len(f1_c)

    precision_o = tp_c.sum().float() / (tp_c + fp_c).sum().float() * 100.0
    recall_o = tp_c.sum().float() / (tp_c + fn_c).sum().float() * 100.0
    f1_o = 2 * precision_o * recall_o / (precision_o + recall_o)

    recall_o = tp_c.sum().float() / (tp_c + fn_c).sum().float() * 100.0
    return mean_p_c, mean_r_c, mean_f_c, precision_o.item(), recall_o.item(), f1_o.item()


class targetCounter:
    def __init__(self) -> None:
        self.num_targets = 0
        self.num_images = 0
        
    def add(self, num_targets=0, num_images=0):
        self.num_targets += num_targets
        self.num_images += num_images
    
    def getAvg(self):
        return self.num_targets / self.num_images
