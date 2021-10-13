from __future__ import division

from models1 import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from PIL import Image

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import scipy.io as io


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    #print(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get dataloader
    dataset = ListDataset(path,img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    i = 1
    sum = 0
    for batch_i, (img_path, imgs, targets) in enumerate(
            tqdm.tqdm(dataloader, desc="Detecting objects")):  # 提取图像和targets
        # sys.stderr.write(str(img_path))
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        contour_maps = detect_contour(img_path, imgs.shape)
        contour_maps = Variable(contour_maps.to(device))
        imgs = Variable(imgs.type(Tensor))


        with torch.no_grad():

            data0 = io.loadmat('plt/outputs_sparse/%d.mat'%i)
            outputs0 = torch.from_numpy(data0['outputs'])
            data1 = io.loadmat('plt/outputs_dense/%d.mat'%i)
            outputs1 = torch.from_numpy(data1['outputs'])
            outputs = torch.cat((outputs0,  outputs1), dim=1)
            i = i + 1
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if bool(list(zip(*sample_metrics))):

        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    else:
        precision = np.zeros((10, 1))
        recall = np.zeros((10, 1))
        AP = np.zeros((10, 1))
        f1 = np.zeros((10, 1))
        ap_class = np.zeros((10, 1))


    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3-F.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/F.data", help="path to data config file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    print("Compute mAP, precision, recall...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=1,
    )
    print("precision:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - precision: {precision[i]}")

    print("recall:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - recall: {recall[i]}")

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
