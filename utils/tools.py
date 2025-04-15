import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import savemat

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type6':
        lr_adjust = {epoch: args.learning_rate * (0.4 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def adjust_learning_rate1(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.model == 'Informer' or args.model == 'DLinear' or args.model == 'Transformer':
        lr_adjust = {epoch: args.learning_rate * (0.75 ** ((epoch - 1) // 1))}
    elif args.model == 'TransformerRPE':
        lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}\n'.format(lr))

def adjust_learning_rate2(optimizer, step, warmup_steps, args):
    step = max(1, step)  # 避免 step = 0 时报错
    if args.model == 'Informer' or args.model == 'DLinear' or args.model == 'Transformer':
        if args.lradj == 'original':
            lr_adjust = {step: (args.d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))}
        elif args.lradj == 'setLR':
            lr_adjust = {step: args.learning_rate * (warmup_steps ** 0.5) * (args.d_model ** 0.5) * (args.d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))}
        elif args.lradj == 'cosine':
            lr_adjust = {step: args.learning_rate / 2 * (1 + math.cos(step / (args.train_epochs * (warmup_steps / args.warmup_epochs) * math.pi)))}
        
    if step in lr_adjust.keys():
        lr = lr_adjust[step]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, epoch, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.8f} --> {val_loss:.8f}).  Saving model ...')
        checkpoits = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': val_loss,
            }
    
        torch.save(checkpoits, path + '/checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def getMeanStd(symbol, flag):
    if flag == 0:
        dataset_path = f'dataset/paperDataset/IQ_CFR_SNR{symbol}.csv'
        data = pd.read_csv(dataset_path, header=0, index_col=0)
    else:
        dataset_path = f"./dataset/china/UMA4Rx32Tx5Ms8RB{symbol}km.npy"
        nd = np.load(dataset_path)
        real = nd[ :, :, 0, :, :, :]
        imag = nd[ :, :, 1, :, :, :]
        value = np.concatenate((real, imag), axis=-1).transpose(0, 2, 1, 3, 4)
        data = pd.DataFrame(value.reshape(-1, value.shape[3] * value.shape[4]))

    num_train = int(len(data) * 0.8)
    num_test = int(len(data) * 0.1)
    num_vali = len(data) - num_train - num_test
    border1s = [0, num_train, len(data) - num_test]
    border2s = [num_train, num_train + num_vali, len(data)]

    # 使用 .iloc 进行基于位置的切片
    train_data = data.iloc[border1s[0]:border2s[0]]
    val_data = data.iloc[border1s[1]:border2s[1]]
    test_data = data.iloc[border1s[2]:border2s[2]]

    # 计算测试集的均值和标准差
    mean = np.mean(test_data, axis=0)  # axis=0 表示沿着列计算统计量
    std = np.std(test_data, axis=0)  # axis=0 表示沿着列计算统计量

    return mean, std, data, num_train, num_vali, num_test


def getRealNMSE(source_path, symbol, num):
    mean, std, data, num_train, num_vali, num_test = getMeanStd(symbol, num)

    pred_path = os.path.join(source_path, "pred.npy")
    true_path = os.path.join(source_path, "true.npy")
    pred = np.load(pred_path, allow_pickle=True)[:, 0, :]
    true = np.load(true_path, allow_pickle=True)[:, 0, :]
    print(pred.shape)

    pred_origin = np.zeros(np.shape(pred))
    true_origin = np.zeros(np.shape(pred))
    for col in range(np.shape(pred)[1]):
        pred_origin[:, col] = pred[:, col] * std[col] + mean[col]
        true_origin[:, col] = true[:, col] * std[col] + mean[col]

    pred_origin = pred_origin.transpose()
    true_origin = true_origin.transpose()

    # 假设 y_true 是真实值矩阵，y_pred 是预测值矩阵
    y_true = true_origin
    y_pred = pred_origin
    savemat(source_path + '/data.mat',{'pred': y_pred, 'true': y_true})

    # 计算 NMSE
    mse = np.mean((y_true - y_pred) ** 2)
    nmse = mse / np.mean(y_true ** 2)

    print("real MSE:", mse)
    print("real NMSE:", nmse)
    return mse, nmse
