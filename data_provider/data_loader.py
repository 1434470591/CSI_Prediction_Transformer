import os
import warnings
import numpy as np

import pandas as pd
import torch
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from utils.timefeatures import time_features
from utils.tools import StandardScaler
from torch.utils.data import Dataset
import torch.utils.data as Data
from torch.autograd import Variable
from test1 import __calculate_china_delta_CFR_first__, __calculate_china_delta_CIR_first__, __calculate_china_delta_CFR_second__, __calculate_china_delta_CIR_second__
from test1 import __calculate_delta_CFR_first__, __calculate_delta_CIR_first__, __calculate_delta_CFR_second__, __calculate_delta_CIR_second__
from test1 import __calculate_deltas_efficient_first__, __calculate_deltas_efficient_second__
warnings.filterwarnings('ignore')

class SJTU_original(Dataset):
    def __init__(self, args, flag="train"):
        self.args = args
        self.flag = flag

        assert flag in [
            "train",
            "test",
            "vali",
            "pred",
        ]

        type_map = {
            "train": 0,
            "vali": 1,
            "test": 2,
            "pred": 2,
        }
        self.set_type = type_map[flag]

        self.seq_len = args.seq_len
        # self.label_len = args.label_len
        self.slid_step = args.slid_step
        self.pred_len = args.pred_len 
        
        self.__load_data__()

    def __load_data__(self):
        """
        TODO: download data
        """
        df_raw = pd.read_csv(self.args.root_path + self.args.data_path, encoding="gbk", index_col=0)
        # df = pd.read_csv("./dataset/" + self.args.data_path, encoding="gbk")
        # columns = df.columns
        df_raw.fillna(df_raw.mean(), inplace=True)
        df = df_raw.values[:, :64]
        if self.args.enhancement:
            magnitude = np.sqrt(df[:, :32] ** 2 + df[:, -32:] ** 2)
            data = np.concatenate((df, magnitude, magnitude ** 2,), axis=1)
        else :
            data = df
        """
        TODO: split train&vali&test
        """
        num_train = int(len(data) * self.args.split_ratio[0])
        num_test = int(len(data) * self.args.split_ratio[2])
        num_vali = len(data) - num_train - num_test
        
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(data)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        """
        TODO: standardization
        """
        if self.args.scale:
            train_data = data[border1s[0] : border2s[0]]
            self.scaler = StandardScaler(train_data)
            data = self.scaler.transform(data=data)

        self.data_x = data[border1: border2]
        self.data_y = data[border1: border2]

        self.data_stamp = [[(i + j + 1) for i in range(data.shape[1])] for j in range(data.shape[0])]
        # print(self.data_stamp)

    def __getitem__(self, index):
        x_begin = index * self.slid_step
        x_end = x_begin + self.seq_len
        # y_begin = x_end - self.label_len
        # y_end = y_begin + self.label_len + self.pred_len
        y_begin = x_end
        y_end = y_begin + self.pred_len

        seq_x = self.data_x[x_begin: x_end]
        seq_y = self.data_y[y_begin: y_end]

        seq_x_mark = self.data_stamp[x_begin: x_end]
        seq_y_mark = self.data_stamp[y_begin: y_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len -self.pred_len + 1
    
    def invers(self, data):
        return self.scaler.inverse_transform(data)

class CHINA():
    def __init__(self, args, flag):
        assert flag in [
            "train",
            "test",
            "vali",
            "pred",
        ]

        type_map = {
            "train": 0,
            "vali": 1,
            "test": 2,
            "pred": 2,
        }
        self.args = args
        self.flag = flag
        self.set_type = type_map[flag]
        
    def __load_data__(self):
        if self.args.scenario == 'MIMO':
            """
            TODO: download data and concatenate real&imaginary part
            Data shape: (UE number, Time, Real&Imaginary parts, Tx, Rx, RB number) shape(21000, 20, 2, 32, 4, 8)
            """
            nd = np.load(f"./dataset/china/{self.args.data_path}")[:5000]
            if self.args.space == 1:
                # TODO:Tx&Rx to RB number:(UE number, Time, Tx, Rx, RB number, Real&Imaginary parts)
                self.UE_number, self.Time, _, self.args.Tx, self.args.Rx, self.RB_number = nd.shape
                value = nd.transpose(0, 1, 3, 4, 5, 2)
                data = value.reshape(value.shape[0], value.shape[1], -1)
                # df = value.reshape(value.shape[0], value.shape[1], -1)
            else:
                # TODO: Rx to feature &Tx to UE number --> (UE number, Tx, Time, Rx, RB number, Real&Imaginary parts,)
                value = nd.transpose(0, 3, 1, 4, 5, 2)
                data = value.reshape(value.shape[0] * value.shape[1], value.shape[2], -1)
        elif self.args.scenario == 'SISO':
            nd = np.load(self.args.root_path + self.args.data_path)[ :, :, :, self.args.Txth, self.args.Rxth, :]
            self.UE_number, self.Time, _, self.RB_number = nd.shape
            CFR = nd[:, :, 0, :] + 1j * nd[:, :, 1, :]
            data = nd.reshape(self.UE_number, self.Time, -1)
        """
        TODO: split train&vali&test
        """
        len = data.shape[0]
        num_train = int(len * self.args.split_ratio[0])
        num_test = int(len * self.args.split_ratio[2])
        num_vali = len - num_train - num_test
        
        border1s = [0, num_train, len - num_test]
        border2s = [num_train, num_train + num_vali, len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        """
        TODO:standardization
        """
        if self.args.scale:
            train_data = data[border1s[0] : border2s[0]]
            self.scaler = StandardScaler(train_data)
            self.args.data_mean, self.args.data_std = self.scaler.mean, self.scaler.std
            data = self.scaler.transform(data=data)
        data = data[border1: border2]
        """
        TODO: slid window
        """
        X = []
        y = []
        X_mark = []
        y_mark = []
        tcVec = []
        for k in tqdm(range(data.shape[0])):
            i = j = 0
            step = self.args.slid_step
            while (i + self.args.seq_len + self.args.pred_len) <= self.Time:
                step_CFR = CFR[k, i : i + self.args.seq_len, ...]
                X.append(data[k, i : i + self.args.seq_len, ...])
                if self.args.features == "MS":
                    y.append(data[k, i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len, self.args.target - 1])
                if self.args.features == "M":
                    y.append(data[k, i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len, ...])
                if self.args.axis == 0 and self.args.useEmbedding:
                    if self.args.method == 'second':
                        CFR_vector, CIR_vectoer = __calculate_china_delta_CFR_second__(step_CFR, self.args.axis), __calculate_china_delta_CIR_second__(step_CFR, self.args.axis)
                        vector = ((__calculate_deltas_efficient_second__(CFR_vector, EmbeddingType=self.args.EmbeddingType, threshold=self.args.threshold)) + (__calculate_deltas_efficient_second__(CFR_vector, EmbeddingType=self.args.EmbeddingType, threshold=self.args.threshold))) / 2 if self.args.EmbeddingResponse=='MIX' else (__calculate_deltas_efficient_second__(Y_vector=CFR_vector if self.args.EmbeddingResponse == 'CFR' else CIR_vectoer, EmbeddingType=self.args.EmbeddingType, threshold=self.args.threshold))
                    elif self.args.method == 'first':
                        CFR_vector_L1, CFR_vector_L2 = __calculate_china_delta_CFR_first__(step_CFR, self.args.axis)
                        CIR_vector_L1, CIR_vector_L2 = __calculate_china_delta_CIR_first__(step_CFR, self.args.axis)
                        if self.args.EmbeddingResponse == 'MIX' :
                            L1vector, L2vector =(__calculate_deltas_efficient_first__(CFR_vector_L1, self.args.threshold) + __calculate_deltas_efficient_first__(CIR_vector_L1, self.args.threshold)) / 2, (__calculate_deltas_efficient_first__(CFR_vector_L2, self.args.hreshold) + __calculate_deltas_efficient_first__(CIR_vector_L2, self.args.threshold)) / 2 
                        else :
                            L1vector, L2vector = __calculate_deltas_efficient_first__(CFR_vector_L1 if self.args.EmbeddingResponse == 'CFR' else CIR_vector_L1, self.args.threshold), __calculate_deltas_efficient_first__(CFR_vector_L2 if self.args.EmbeddingResponse == 'CFR' else CIR_vector_L2, self.args.threshold)
                        vector = L1vector if self.args.EmbeddingType == 'L1' else L2vector
                    tcVec.append(vector)
                else:
                    X_mark.append(range( i + 1 + k * self.Time, i + self.args.seq_len + 1 + k * self.Time))
                    y_mark.append(range( i + self.args.seq_len + 1 + k * self.Time, i + self.args.seq_len + self.args.pred_len + 1 + k * self.Time))
                i += step
                j += 1        
        X, y, X_mark, y_mark = np.array(X), np.array(y), np.array(X_mark), np.array(y_mark)
        tcVec = np.array(tcVec)
        """
        TODO: convert to torch format
        """
        self.data_x = Variable(torch.Tensor(X))
        self.data_y = Variable(torch.Tensor(y))

        self.data_x_mark = Variable(torch.Tensor(X_mark))
        self.data_y_mark = Variable(torch.Tensor(y_mark))
        
        if self.args.useEmbedding and self.args.axis ==0 :
            return Data.TensorDataset(self.data_x, self.data_y, Variable(torch.Tensor(tcVec)), Variable(torch.Tensor(tcVec)))
        return Data.TensorDataset(self.data_x, self.data_y, self.data_x_mark, self.data_y_mark)
            

class SJTU():
    def __init__(self, args, flag):
        assert flag in [
            "train",
            "test",
            "vali",
            "pred",
        ]

        type_map = {
            "train": 0,
            "vali": 1,
            "test": 2,
            "pred": 2,
        }
        self.args = args
        self.flag = flag
        self.set_type = type_map[flag]

    def __load_data__(self):
        """
        TODO: download data
        """
        df = pd.read_csv(self.args.root_path + self.args.data_path, encoding="gbk", index_col=0)
        # df = pd.read_csv("./dataset/" + self.args.data_path, encoding="gbk")
        columns = df.columns
        df.fillna(df.mean(), inplace=True)
        data = df.values[:, :64]
        if self.args.enhancement:
            magnitude = np.sqrt(data[:, :32] ** 2 + data[:, -32:] ** 2)
            data = np.concat((data, magnitude, magnitude ** 2, ), axis=1)
        CFR = data[:, :32] + 1j * data[:, -32:]
        """
        TODO: split train&vali&test and convert to torch format
        """
        len = data.shape[0] 
        num_train = int(len * self.args.split_ratio[0])
        num_test = int(len * self.args.split_ratio[2])
        num_vali = len - num_train - num_test

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        """
        TODO: standardization
        """
        if self.args.scale:
            train_data = data[border1s[0]: border2s[0]]
            scaler = StandardScaler(train_data)
            data = scaler.transform(data)
        
        data = data[border1: border2]
        """
        TODO: slid window
        """
        i = j = 0
        X = []
        y = []
        tcVec = []
        X_mark = []
        y_mark = []
        step = self.args.slid_step
        while (i + self.args.seq_len + self.args.pred_len) <= data.shape[0]:
            step_data = CFR[i : i + self.args.seq_len, :]
            X.append(data[i : i + self.args.seq_len, :])
            if self.args.features == "MS":
                y.append(data[i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len, self.args.target - 1])
            if self.args.features == "M":
                y.append(data[i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len, :])
            if self.args.axis == 0 and self.args.useEmbedding:
                if self.args.method == 'first':
                    CFR_vector_L1, CFR_vector_L2 = __calculate_delta_CFR_first__(axis=self.args.axis, CFR=step_data, load=False)
                    CIR_vector_L1, CIR_vector_L2 = __calculate_delta_CIR_first__(axis=self.args.axis, CFR=step_data, load=False)
                    if self.args.EmbeddingResponse == 'MIX' :
                        L1vector, L2vector =(__calculate_deltas_efficient_first__(CFR_vector_L1, self.args.threshold) + __calculate_deltas_efficient_first__(CIR_vector_L1, self.args.threshold)) / 2, (__calculate_deltas_efficient_first__(CFR_vector_L2, self.args.threshold) + __calculate_deltas_efficient_first__(CIR_vector_L2, self.args.threshold)) / 2 
                    else :
                        L1vector, L2vector = __calculate_deltas_efficient_first__(CFR_vector_L1 if self.args.EmbeddingResponse == 'CFR' else CIR_vector_L1, self.args.threshold), __calculate_deltas_efficient_first__(CFR_vector_L2 if self.args.EmbeddingResponse == 'CFR' else CIR_vector_L2, self.args.threshold)
                    vector = L1vector if self.args.EmbeddingType == 'L1' else L2vector
                elif self.args.method == 'second':
                    CFR_vector, CIR_vector = __calculate_delta_CFR_second__(axis=self.args.axis, load=False, CFR=step_data), __calculate_delta_CIR_second__(axis=self.args.axis, load=False, CFR=step_data)
                    vector =(__calculate_deltas_efficient_second__(CFR_vector, EmbeddingType=self.args.EmbeddingType, threshold=self.args.threshold) + __calculate_deltas_efficient_second__(CIR_vector, EmbeddingType=self.args.EmbeddingType, threshold=self.args.threshold)) / 2 if self.args.EmbeddingResponse == 'MIX' else __calculate_deltas_efficient_second__(Y_vector=CFR_vector if self.args.EmbeddingResponse == 'CFR' else CIR_vector, EmbeddingType=self.args.EmbeddingType, threshold=self.args.threshold) 
                tcVec.append(vector)
            else:
                X_mark.append(range(i + 1, i + self.args.seq_len+1))
                y_mark.append(range(i + self.args.seq_len + 1, i + self.args.seq_len + self.args.pred_len + 1))
            i += step
            j += 1
        X, y, X_mark, y_mark = np.array(X), np.array(y), np.array(X_mark), np.array(y_mark)
        tcVec = np.array(tcVec)
        self.data_x = Variable(torch.Tensor(X))
        self.data_y = Variable(torch.Tensor(y))

        self.data_x_mark = Variable(torch.Tensor(X_mark))
        self.data_y_mark = Variable(torch.Tensor(y_mark))

        if self.args.axis == 0 and self.args.useEmbedding:
            return  Data.TensorDataset(self.data_x, self.data_y, Variable(torch.Tensor(tcVec)), Variable(torch.Tensor(tcVec)))
        
        return  Data.TensorDataset(self.data_x, self.data_y, self.data_x_mark, self.data_y_mark)


# class SJTU():
#     def __init__(self, args, flag):
#         assert flag in [
#             "train",
#             "test",
#             "vali",
#         ]

#         type_map = {
#             "train": 0,
#             "vali": 1,
#             "test": 2,
#         }
#         self.args = args
#         self.flag = flag
#         self.set_type = type_map[flag]

#     def __load_data__(self):
#         """
#         TODO: download data
#         """
#         df = pd.read_csv(self.args.root_path + self.args.data_path, encoding="gbk", index_col=0)
#         # df = pd.read_csv("./dataset/" + self.args.data_path, encoding="gbk")
#         columns = df.columns
#         df.fillna(df.mean(), inplace=True)
#         data = df.values[:, :64]
#         if self.args.enhancement:
#             magnitude = np.sqrt(data[:, :32] ** 2 + data[:, -32:] ** 2)
#             data = np.concat((data, magnitude, magnitude ** 2, ), axis=1)
       
#         """
#         TODO: split train&vali&test and convert to torch format
#         """
#         len = data.shape[0] 
#         num_train = int(len * self.args.split_ratio[0])
#         num_test = int(len * self.args.split_ratio[2])
#         num_vali = len - num_train - num_test

#         border1s = [0, num_train, num_train + num_vali]
#         border2s = [num_train, num_train + num_vali, len]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#         """
#         TODO: standardization
#         """
#         if self.args.scale:
#             train_data = data[border1s[0]: border2s[0]]
#             scaler = StandardScaler(train_data)
#             data = scaler.transform(data)
        
#         data = data[border1: border2]
#         """
#         TODO: slid window
#         """
#         i = j = 0
#         X = []
#         y = []
#         X_mark = []
#         y_mark = []
#         step = self.args.slid_step
#         while (i + self.args.seq_len + self.args.pred_len) <= data.shape[0]:
#             X.append(data[i : i + self.args.seq_len, :])
#             if self.args.features == "MS":
#                 y.append(data[i + self.args.seq_len : i + self.args.seq_len + self.args.pred_len, self.args.target - 1])
#             if self.args.features == "M":
#                 y.append(data[i + self.args.seq_len : i + self.args.seq_len + self.args.pred_len, :])

#             X_mark.append(range(i + 1, i + self.args.seq_len+1))
#             y_mark.append(range(i + self.args.seq_len + 1, i + self.args.seq_len + self.args.pred_len + 1))
#             i += step
#             j += 1
#         X, y, X_mark, y_mark = np.array(X), np.array(y), np.array(X_mark), np.array(y_mark)

#         self.data_x = Variable(torch.Tensor(X))
#         self.data_y = Variable(torch.Tensor(y))

#         self.data_x_mark = Variable(torch.Tensor(X_mark))
#         self.data_y_mark = Variable(torch.Tensor(y_mark))
        
#         return  Data.TensorDataset(self.data_x, self.data_y, self.data_x_mark, self.data_y_mark)

# class CHINA():
#     def __init__(self, args, flag):
#         assert flag in [
#             "train",
#             "test",
#             "vali",
#         ]

#         type_map = {
#             "train": 0,
#             "vali": 1,
#             "test": 2,
#         }
#         self.args = args
#         self.flag = flag
#         self.set_type = type_map[flag]
        
#     def __load_data__(self):
#         if self.args.scenario == 'MIMO':
#             """
#             TODO: download data and concatenate real&imaginary part
#             Data shape: (UE number, Time, Real&Imaginary parts, Tx, Rx, RB number) shape(21000, 20, 2, 32, 4, 8)
#             """
#             nd = np.load(f"./dataset/china/{self.args.data_path}")[:5000]
#             if self.args.space == 1:
#                 # TODO:Tx&Rx to RB number:(UE number, Time, Tx, Rx, RB number, Real&Imaginary parts)
#                 self.UE_number, self.Time, _, self.args.Tx, self.args.Rx, self.RB_number = nd.shape
#                 value = nd.transpose(0, 1, 3, 4, 5, 2)
#                 data = value.reshape(value.shape[0], value.shape[1], -1)
#                 # df = value.reshape(value.shape[0], value.shape[1], -1)
#             else:
#                 # TODO: Rx to feature &Tx to UE number --> (UE number, Tx, Time, Rx, RB number, Real&Imaginary parts,)
#                 value = nd.transpose(0, 3, 1, 4, 5, 2)
#                 data = value.reshape(value.shape[0] * value.shape[1], value.shape[2], -1)
        
#         elif self.args.scenario == 'SISO':
#             nd = np.load(self.args.root_path + self.args.data_path)[:, :, :, self.args.Txth, self.args.Rxth, :]
#             self.UE_number, self.Time, _, self.RB_number = nd.shape
#             data = nd.reshape(self.UE_number, self.Time, -1)
#         """
#         TODO: split train&vali&test
#         """
#         len = data.shape[0]
#         num_train = int(len * self.args.split_ratio[0])
#         num_test = int(len * self.args.split_ratio[2])
#         num_vali = len - num_train - num_test
        
#         border1s = [0, num_train, len - num_test]
#         border2s = [num_train, num_train + num_vali, len]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]

#         """
#         TODO:standardization
#         """
#         if self.args.scale:
#             train_data = data[border1s[0] : border2s[0]]
#             self.scaler = StandardScaler(train_data)
#             data = self.scaler.transform(data=data)

#         data = data[border1: border2]

#         """
#         TODO: slid window
#         """
#         i = j = 0
#         X = []
#         y = []
#         a = []
#         b = []
#         step = self.args.slid_step

#         while (i + self.args.seq_len + self.args.pred_len) <= data.shape[1]:
#             X.append(data[:, i : i + self.args.seq_len, ...])
#             if self.args.features == "MS":
#                 y.append(data[:, i + self.args.seq_len - self.args.label_len : i + self.args.seq_len + self.args.pred_len, self.args.target - 1])
#             if self.args.features == "M":
#                 y.append(data[:, i + self.args.seq_len - self.args.label_len : i + self.args.seq_len + self.args.pred_len, ...])
#             # if self.args.features == "MS":
#             #     y.append(data[:, i : i + self.args.seq_len + self.args.pred_len, self.args.target - 1])
#             # if self.args.features == "M":
#             #     y.append(data[:, i : i + self.args.seq_len + self.args.pred_len, ...])
#             i += step
#             j += 1
#             a.append(range( i + 1, i + self.args.seq_len+1))
#             b.append(range( i + self.args.seq_len + 1, i + self.args.seq_len + self.args.pred_len + 1))
        
#         # [0, 20)
#         # X_mark, y_mark = np.array(a), np.array(b)
#         # X_mark, y_mark = np.tile(X_mark, reps=(data.shape[0], 1)), np.tile(y_mark, reps=(data.shape[0], 1))

#         X, y = np.array(X).transpose(1, 0, 2, 3), np.array(y).transpose(1, 0, 2, 3)
#         X, y = X.reshape((-1,) + X.shape[2:]), y.reshape((-1,) + y.shape[2:])
#         print(f"------------------------------Get {self.flag} mark------------------------------")
#         X_mark = []
#         y_mark = []
#         for k in tqdm(range(data.shape[0])):
#             i = 0
#             j = 0
#             step = self.args.slid_step
#             while (i + self.args.seq_len + self.args.pred_len) <= data.shape[1]:
#                 X_mark.append(range( i + 1 + k * 20, i + self.args.seq_len + 1 + k * 20))
#                 y_mark.append(range( i + self.args.seq_len - self.args.label_len + 1 + k * 20, i + self.args.seq_len + self.args.pred_len + 1 + k * 20))
#                 i += step
#                 j += 1
#         X_mark, y_mark = np.array(X_mark), np.array(y_mark)
#         """
#         TODO: convert to torch format
#         """
#         self.data_x = Variable(torch.Tensor(X))
#         self.data_y = Variable(torch.Tensor(y))

#         self.data_x_mark = Variable(torch.Tensor(X_mark))
#         self.data_y_mark = Variable(torch.Tensor(y_mark))

#         return Data.TensorDataset(self.data_x, self.data_y, self.data_x_mark, self.data_y_mark)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'groundtruth_test', 'groundtruth_train', 'groundtruth_val']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'groundtruth_train': 0, 'groundtruth_val': 1,
                    'groundtruth_test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # 元勋
        # self.scaler = StandardScaler()
        # 伯涛
        # self.scaler = MinMaxScaler((-1.5, 1.5))
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # todo 此处分开处理实部虚部

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        time_name = 'time'
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(time_name)
        df_raw = df_raw[[time_name] + cols + [self.target]]
        # 拆分数据集
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]

            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[time_name]][border1:border2]
        # df_stamp[time_name] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_stamp = torch.tensor([int(x[0].replace('snapshot', '')) for x in self.data_stamp], dtype=torch.int64)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        time_name = 'time'
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove(time_name)
        df_raw = df_raw[[time_name] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[[time_name]][border1:border2]
        # tmp_stamp[time_name] = pd.to_datetime(tmp_stamp.date)
        # pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
        pred_dates = []
        print(tmp_stamp.size, len(tmp_stamp), tmp_stamp.shape, type(tmp_stamp))
        start_object = tmp_stamp[time_name].iloc[-1]
        start_index = int(start_object[8:])
        for i in range(self.pred_len):
            pred_dates.append('Snapshot{}'.format(start_index + i + 1))

        df_stamp = pd.DataFrame(columns=[time_name])
        # df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        df_stamp[time_name] = list(tmp_stamp[time_name].values) + list(pred_dates[1:])
        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop([time_name], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
