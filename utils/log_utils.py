import codecs
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def get_timestamp(format='%Y-%m-%d %H:%M:%S'):
    return datetime.strftime(datetime.now(), format)


def info(msg):
    print('[%s]%s' % (get_timestamp(), msg))


class ClassifyLogger:
    def __init__(self, data_path='', log_file='log.txt', plot_file='plot.png'):
        self.data_path = data_path
        self.log_file = log_file
        self.plot_file = plot_file

        if self.data_path != '':
            with codecs.open('%s/%s' % (self.data_path, self.log_file), 'w', 'utf-8') as fout:
                fout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (
                    'time', 'epoch', 'train loss', 'train accuracy',
                    'test accuracy', 'test precision', 'test recall', 'test f1', 'remark'))

    def write(self, epoch, train_loss, train_acc, test_acc, test_pre, test_rec, test_f1, remark=''):
        with codecs.open('%s/%s' % (self.data_path, self.log_file), 'a', 'utf-8') as fout:
            fout.write('%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\n' % (
                get_timestamp(), epoch, train_loss, train_acc, test_acc, test_pre, test_rec, test_f1, remark))

    def draw_plot(self, data_path='', log_file='', plot_file=''):
        if data_path == '':
            data_path = self.data_path
        if log_file == '':
            log_file = self.log_file
        if plot_file == '':
            plot_file = self.plot_file

        eppch = []
        train_loss = []
        train_acc = []
        test_acc = []
        test_pre = []
        test_rec = []
        test_f1 = []

        with codecs.open('%s/%s' % (data_path, log_file), 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                eppch.append(int(line[1]) - 1)
                train_loss.append(float(line[2]))
                train_acc.append(float(line[3]))
                test_acc.append(float(line[4]))
                test_pre.append(float(line[5]))
                test_rec.append(float(line[6]))
                test_f1.append(float(line[7]))

        # x_locator = MultipleLocator(int(len(eppch) / 5))
        # y_locator = MultipleLocator(int(len(eppch) / 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        ax = plt.subplot2grid((2, 4), (0, 0), title='train loss', colspan=2)
        # ax.xaxis.set_major_locator(x_locator)
        # ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, train_loss)

        ax = plt.subplot2grid((2, 4), (0, 2), title='train accuracy', colspan=2)
        # ax.xaxis.set_major_locator(x_locator)
        # ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, train_acc)

        ax = plt.subplot2grid((2, 4), (1, 0), title='test accuracy')
        # ax.xaxis.set_major_locator(x_locator)
        # ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, test_acc)

        ax = plt.subplot2grid((2, 4), (1, 1), title='test precision')
        # ax.xaxis.set_major_locator(x_locator)
        # ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, test_pre)

        ax = plt.subplot2grid((2, 4), (1, 2), title='test recall')
        # ax.xaxis.set_major_locator(x_locator)
        # ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, test_rec)

        ax = plt.subplot2grid((2, 4), (1, 3), title='test f1')
        # ax.xaxis.set_major_locator(x_locator)
        # ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, test_f1)

        plt.rcParams['savefig.dpi'] = 300
        plt.savefig('%s/%s' % (data_path, plot_file))
