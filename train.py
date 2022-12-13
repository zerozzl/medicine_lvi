import os
import codecs
import logging
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_loader import ImageDataset, load_data
from models.VanillaCNN import CNN
from models.VggNet import VggNet
from models.ResNet import ResNet18
import evaluator
from utils import torch_utils, log_utils


def get_dataset(args, pixels, labels, train_ids, test_ids):
    train_pixels = {key: pixels[key] for key in train_ids}
    train_labels = {key: labels[key] for key in train_ids}
    test_pixels = {key: pixels[key] for key in test_ids}
    test_labels = {key: labels[key] for key in test_ids}

    train_dataset = ImageDataset(train_pixels, train_labels, args.input_size,
                                 input_norm=args.input_norm, do_train=True, debug=args.debug)
    test_dataset = ImageDataset(test_pixels, test_labels, args.input_size,
                                input_norm=args.input_norm, do_train=False, debug=args.debug)

    return train_dataset, test_dataset


def get_dataloader(args, train_dataset, test_dataset):
    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      sampler=DistributedSampler(train_dataset, shuffle=True))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     sampler=DistributedSampler(test_dataset, shuffle=False))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def build_model(args):
    if args.backbone == 'cnn':
        model = CNN(1, args.classes_num)
    elif args.backbone == 'vggnet':
        model = VggNet(1, args.classes_num)
    elif args.backbone == 'resnet18':
        model = ResNet18(1, args.classes_num)
    return model


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    loss_sum = 0
    correct_sum = 0
    model.train()
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()

        pixels, labels = data
        pixels = pixels.cpu() if args.use_cpu else pixels.cuda()
        labels = labels.cpu() if args.use_cpu else labels.cuda()

        logits = model(pixels)
        preds = torch.argmax(logits, dim=1)

        loss = F.cross_entropy(logits, labels)
        loss = loss.mean()
        loss_sum += loss.item()

        correct = (preds == labels)
        correct_sum += correct.sum().item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    correct_sum = correct_sum / len(dataset)
    return loss_sum, correct_sum


def evaluate(args, dataloader, model, output_path):
    pred_answers = []
    gold_answers = []

    model.eval()
    if args.multi_gpu:
        model = model.module

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            pixels, labels = data
            pixels = pixels.cpu() if args.use_cpu else pixels.cuda()

            logits = model(pixels)
            preds = torch.argmax(logits, dim=1)

            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            gold_answers.extend(labels)
            pred_answers.extend(preds)

    acc, pre, rec, f1 = evaluator.evaluate(gold_answers, pred_answers)
    plot_confusion_matrix(output_path, gold_answers, pred_answers, labels=['阴性', '阳性'])

    return acc, pre, rec, f1


def run_kfold(fi, train_dataset, test_dataset):
    logging.info('========== begin %s fold ==========' % fi)
    output_path = '%s/%s/f%s' % (args.output_path, args.backbone, fi)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = log_utils.ClassifyLogger(data_path=output_path, log_file='train.log', plot_file='train.png')

    logging.info('loading dataset')
    train_dataloader, test_dataloader = get_dataloader(args, train_dataset, test_dataset)

    best_metric = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info('loading pretrained model')
        model, optimizer, epoch, best_metric = torch_utils.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info('creating model')
        model = build_model(args)
        model = model.cpu() if args.use_cpu else model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info('begin training')
    while epoch < args.epoch_size:
        epoch += 1

        train_loss, train_acc = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)
        test_acc, test_pre, test_rec, test_f1 = evaluate(args, test_dataloader, model, output_path)

        logging.info('epoch[%s/%s], train loss: %s, train accuracy: %s' % (
            epoch, args.epoch_size, train_loss, train_acc))
        logging.info('epoch[%s/%s], test accuracy: %s, precision: %s, recall: %s, f1:%s' % (
            epoch, args.epoch_size, test_acc, test_pre, test_rec, test_f1))

        remark = ''
        if test_acc > best_metric:
            best_metric = test_acc
            remark = 'best'
            torch_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_metric)

        torch_utils.save(output_path, 'last.pth', model, optimizer, epoch, best_metric)
        model_logger.write(epoch, train_loss, train_acc, test_acc, test_pre, test_rec, test_f1, remark)

    model_logger.draw_plot()


def plot_confusion_matrix(output_path, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm_norm * 100
    cm_norm = np.around(cm_norm, decimals=2)

    plt.matshow(cm_norm, cmap=plt.cm.Reds)
    # plt.colorbar()

    for i in range(len(cm_norm)):
        for j in range(len(cm_norm)):
            plt.annotate('%s' % cm_norm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
    font_label = FontProperties(fname='./data/fonts/SimHei.ttf', size=13)
    plt.ylabel('实际类别', fontproperties=font_label)
    plt.xlabel('预测类别', fontproperties=font_label)

    font_ticks = FontProperties(fname='./data/fonts/SimHei.ttf', size=11)
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, fontproperties=font_ticks)
    plt.yticks(xlocations, labels, fontproperties=font_ticks)

    # plt.show()
    plt.savefig('%s/confusion_matrix.png' % output_path)
    plt.close('all')


def plot_folds(data_path, kfold_num,
               colors=['#13448c', '#60acdd', '#00a33c', '#a0d193', '#4160ea',
                       '#ff6501', '#9a231b', '#9759b0', '#432a0e', '#494a26']):
    train_loss = {}
    train_accuracy = {}
    test_accuracy = {}
    test_precision = {}
    test_recall = {}
    test_f1 = {}
    for fi in range(kfold_num):
        train_loss['f%s' % fi] = []
        train_accuracy['f%s' % fi] = []
        test_accuracy['f%s' % fi] = []
        test_precision['f%s' % fi] = []
        test_recall['f%s' % fi] = []
        test_f1['f%s' % fi] = []
        with codecs.open('%s/f%s/train.log' % (data_path, fi), 'r', 'utf-8') as fin:
            fin.readline()
            for line in fin:
                line = line.split('\t')
                train_loss['f%s' % fi].append(float(line[2]))
                train_accuracy['f%s' % fi].append(float(line[3]))
                test_accuracy['f%s' % fi].append(float(line[4]))
                test_precision['f%s' % fi].append(float(line[5]))
                test_recall['f%s' % fi].append(float(line[6]))
                test_f1['f%s' % fi].append(float(line[7]))

    iters = list(range(len(train_loss['f0'])))

    color_idx = 0
    plt.figure()
    for fi in train_loss:
        plt.plot(iters, train_loss[fi], label=fi, color=colors[color_idx])
        color_idx += 1
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(0.9, 0.6))
    plt.savefig('%s/train_loss.png' % data_path)
    plt.close('all')

    color_idx = 0
    plt.figure()
    for fi in train_accuracy:
        plt.plot(iters, train_accuracy[fi], label=fi, color=colors[color_idx])
        color_idx += 1
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(bbox_to_anchor=(0.9, 0.6))
    plt.savefig('%s/train_accuracy.png' % data_path)
    plt.close('all')

    color_idx = 0
    plt.figure()
    for fi in test_accuracy:
        plt.plot(iters, test_accuracy[fi], label=fi, color=colors[color_idx])
        color_idx += 1
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(bbox_to_anchor=(0.9, 0.6))
    plt.savefig('%s/test_accuracy.png' % data_path)
    plt.close('all')

    color_idx = 0
    plt.figure()
    for fi in test_precision:
        plt.plot(iters, test_precision[fi], label=fi, color=colors[color_idx])
        color_idx += 1
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.legend(bbox_to_anchor=(0.9, 0.6))
    plt.savefig('%s/test_precision.png' % data_path)
    plt.close('all')

    color_idx = 0
    plt.figure()
    for fi in test_recall:
        plt.plot(iters, test_recall[fi], label=fi, color=colors[color_idx])
        color_idx += 1
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.legend(bbox_to_anchor=(0.9, 0.6))
    plt.savefig('%s/test_recall.png' % data_path)
    plt.close('all')

    color_idx = 0
    plt.figure()
    for fi in test_f1:
        plt.plot(iters, test_f1[fi], label=fi, color=colors[color_idx])
        color_idx += 1
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('f1')
    plt.legend(bbox_to_anchor=(0.9, 0.6))
    plt.savefig('%s/test_f1.png' % data_path)
    plt.close('all')


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch_utils.setup_seed(0)

    if args.multi_gpu:
        logging.info('run on multi GPU')
        torch.distributed.init_process_group(backend='nccl')

    pixels, labels = load_data(args.data_path)
    person_ids = np.array([key for key in pixels])
    person_labels = np.array([labels[key] for key in person_ids])

    kfold = StratifiedKFold(n_splits=args.kfold_num, shuffle=True, random_state=0)
    for fi, (train, test) in enumerate(kfold.split(person_ids, person_labels)):
        train_ids = person_ids[train]
        test_ids = person_ids[test]
        train_dataset, test_dataset = get_dataset(args, pixels, labels, train_ids, test_ids)
        run_kfold(fi, train_dataset, test_dataset)

    plot_folds('%s/%s' % (args.output_path, args.backbone), args.kfold_num)
    logging.info('complete training')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='./data/')
    parser.add_argument('--output_path', type=str,
                        default='./runtime/')
    parser.add_argument('--backbone', type=str,
                        choices=['cnn', 'vggnet', 'resnet18'],
                        default='cnn')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--input_size', type=int,
                        default=64)
    parser.add_argument('--classes_num', type=int,
                        default=2)
    parser.add_argument('--input_norm', type=bool,
                        default=False)
    parser.add_argument('--kfold_num', type=int,
                        default=10)
    parser.add_argument('--epoch_size', type=int,
                        default=100)
    parser.add_argument('--batch_size', type=int,
                        default=32)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-3)
    parser.add_argument('--lr_warmup_proportion', type=float,
                        default=0.1)
    parser.add_argument('--lr_decay_gamma', type=float,
                        default=0.9)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--multi_gpu', type=bool,
                        help='run with: -m torch.distributed.launch',
                        default=False)
    parser.add_argument('--local_rank', type=int,
                        default=0)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
