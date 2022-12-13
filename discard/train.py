import os
import logging
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_loader import ImageDataset
from models.VggNet import VggNet
from models.ResNet import ResNet18
from utils import torch_utils, log_utils


def get_dataset(args, data_path):
    train_dataset = ImageDataset('%s/train' % data_path, args.image_size, do_train=True, debug=args.debug)
    # test_dataset = ImageDataset('%s/test' % data_path, args.image_size, do_train=False, debug=args.debug)
    return train_dataset


def get_dataloader(args, train_dataset):
    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      sampler=DistributedSampler(train_dataset, shuffle=True))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader


def build_model(args):
    if args.backbone == 'vggnet':
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

        labels, images = data
        labels = labels.cpu() if args.use_cpu else labels.cuda()
        images = images.cpu() if args.use_cpu else images.cuda()

        logits = model(images)
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


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch_utils.setup_seed(0)

    if args.multi_gpu:
        logging.info('run on multi GPU')
        torch.distributed.init_process_group(backend='nccl')

    output_path = '%s/%s' % (args.output_path, args.backbone)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = log_utils.ClassifyLogger(data_path=output_path, log_file='train.log', plot_file='train.png')

    logging.info('loading dataset')
    train_dataset = get_dataset(args, args.data_path)
    train_dataloader = get_dataloader(args, train_dataset)

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
        # test_acc = evaluate(args, test_dataloader, model)
        test_acc, test_pre, test_rec, test_f1 = 0, 0, 0, 0

        logging.info('epoch[%s/%s], train loss: %s, train accuracy: %s' % (
            epoch, args.epoch_size, train_loss, train_acc))
        # logging.info('epoch[%s/%s], test accuracy: %s' % (epoch, args.epoch_size, test_acc))

        remark = ''
        if test_acc > best_metric:
            best_metric = test_acc
            remark = 'best'
            torch_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_metric)

        torch_utils.save(output_path, 'last.pth', model, optimizer, epoch, best_metric)
        model_logger.write(epoch, train_loss, train_acc, test_acc, test_pre, test_rec, test_f1, remark)

    model_logger.draw_plot()
    logging.info('complete training')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='./data/effective/')
    parser.add_argument('--output_path', type=str,
                        default='./runtime/')
    parser.add_argument('--backbone', type=str,
                        choices=['vggnet', 'resnet18'],
                        default='vggnet')
    parser.add_argument('--pretrained_backbone_path', type=str,
                        default='./data/pretrained')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--image_size', type=int,
                        default=448)
    parser.add_argument('--classes_num', type=int,
                        default=2)
    parser.add_argument('--batch_size', type=int,
                        default=16)
    parser.add_argument('--epoch_size', type=int,
                        default=100)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-5)
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
                        default=True)

    args = parser.parse_args()

    main(args)
