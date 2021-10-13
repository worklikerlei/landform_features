



from __future__ import division


from models_dense import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy


def detect_contour(img_path, shape):
    contour_maps = np.zeros((shape[0], 1, shape[2], shape[3]))
    # print(contour_maps.shape)
    count = 0
    for path in img_path:
        img = Image.open(path)
        img = img.transpose(Image.ROTATE_180)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.resize((shape[2], shape[3]), Image.BILINEAR)
        img = np.array(img)
        # img=img[...,0]
        xx = np.arange(0, img.shape[1], 1)
        yy = np.arange(0, img.shape[0], 1)
        pos_x, pos_y = np.meshgrid(xx, yy)
        cntr = plt.contour(pos_x, pos_y, img, 15)
        img_c = np.zeros(img.shape)
        coordinates = cntr.allsegs
        for i in range(len(cntr.layers)):
            coor_i = coordinates[i]
            for j in range(len(coor_i)):
                # print('shape of segment {} = {}'.format(j, coor_i[j].shape))
                pos = np.array(coor_i[j], dtype=np.int)
                img_c[pos[:, 1], pos[:, 0]] = cntr.layers[i]
        if np.max(img_c):
            img_c = img_c / np.max(img_c)
        else:
            img_c = img_c
        contour_maps[count, 0, :, :] = img_c

        count = count + 1
    contour_maps = torch.from_numpy(contour_maps)
    # print(img_c.shape)
    return contour_maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-F.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/F.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, default="weights/weights_dense.pth",
                        help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]

    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, img_size=opt.img_size, augment=True, multiscale=opt.multiscale_training)
    # print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        b = []
        loss_try = []
        model.train()
        # print(model)
        start_time = time.time()
        for batch_i, (img_path, imgs, targets) in enumerate(dataloader):
            # print(contour_maps.shape)

            # for path in img_path:
            contour_maps = detect_contour(img_path, imgs.shape)
            #    fig, ax = plt.subplots(1)
            #    ax.imshow(img)
            #    plt.axis("off")
            #    plt.show()
            #    plt.close()
            b.append(batch_i)
            batches_done = len(dataloader) * epoch + batch_i
            # input n*images output n1*contours
            # contour_maps = detect_contour(images)
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            contour_maps = Variable(contour_maps.to(device))
            # loss, outputs = model(imgs, targets,contour_maps)
            loss, outputs = model(imgs, contour_maps, targets)
            del (contour_maps)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"
            loss_try.append(loss.item())

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.001,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP", "precision", "recall"]]
            for i, c in enumerate(ap_class):
                print(c, class_names[c])
                ap_table += [[c, class_names[c], "%.5f" % AP[i], "%.5f" % precision[i], "%.5f" % recall[i]]]

            print(AsciiTable(ap_table).table)
            print(f"---- precision {precision.mean()}")
            print(f"---- recall {recall.mean()}")
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"/checkpoints/dense_%d.pth" % epoch)
