import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from scipy.spatial.distance import cosine

def get_vector_modulus(x, y):
    return sqrt(x ** 2 + y ** 2)


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def NMSE(pred, true):
    return np.mean((pred - true) ** 2) / (np.mean(true ** 2) + 1e-10)

def Adjusted_R2(pred, true, p):
    """
    计算调整后的 R^2 (Adjusted R^2)
    
    :param y_true: 真实值 (numpy array)
    :param y_pred: 预测值 (numpy array)
    :param p: 自变量的数量
    :return: 调整后的 R^2
    """
    n = len(true)  # 样本数
    r2 = r2_score(true, pred)  # 计算普通的 R^2
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)  # 计算调整后的 R^2
    return adj_r2

def SGCS(preds, trues):
    sgcs_list = []
    channel_size = int(preds.shape[2] / 2)
    for i in range(preds.shape[0]):
        for timestamp in range(preds.shape[1]):
            for feature_index in range(channel_size):
                a_x = preds[i, timestamp, feature_index]
                a_y = preds[i, timestamp, feature_index + channel_size]
                b_x = trues[i, timestamp, feature_index]
                b_y = trues[i, timestamp, feature_index + channel_size]
                a, b = [a_x, a_y], [b_x, b_y]
                cosine_similarity = 1 - cosine(a, b)
                sgcs_list.append(cosine_similarity)
    SGCS = np.average(sgcs_list)
    return SGCS 
def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    mape = MAPE(pred, true)
    rmse = RMSE(pred, true)
    mspe = MSPE(pred, true)
    nmse = NMSE(pred, true)
    nmse_db = 10 * np.log10(nmse)
    # adjusted_r2 = Adjusted_R2(pred, true, pred.shape[-1])
    sgcs = SGCS(pred, true)
    return mae, mse, rmse, mape, mspe, nmse, nmse_db, sgcs
