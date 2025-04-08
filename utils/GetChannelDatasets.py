import h5py
from scipy.interpolate import interp1d
import scipy.io
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def loadCIR():
    # 从.mat文件中读取复数矩阵CIR
    mat_data = scipy.io.loadmat("../dataset/paperDataset/data_wo_noise/test.mat")
    CIR = mat_data['sum_CIR']
    gateChangeRatio = [0.0915, 0.011, 0.0109, 0.0076]  # [ΔA, Δtheta, ΔI, ΔQ]

    # 对只包含确定性移动的部分，60km/h的运动速度，共10s,1000sample/s，全程运动情况下，
    # IQ的real_part，imag_part在9snapshot内的变化量绝对值的平均值为[0.09148969348668239, 0.011013250188858473]
    # Atheta的A,theta在9snapshot内的变化量绝对值的平均值为[0.010929250449998236, 0.0075608610017994]

    # 找到包含值为"nan"的列
    nan_cols = np.any(np.isnan(CIR), axis=0)
    # 删除包含值为"nan"的列
    CIR = CIR[:, ~nan_cols]
    CIR = np.transpose(CIR)
    # 拆分复数矩阵CIR为实部矩阵realPart和虚部矩阵imagPart
    realPart = np.real(CIR)
    imagPart = np.imag(CIR)

    # 得到连接矩阵IQ=[realPart, imagPart]
    IQ = np.concatenate((realPart, imagPart), axis=1)

    # 计算幅值矩阵power和相位矩阵phase
    power = np.abs(CIR)
    phase = np.angle(CIR)

    # 连接幅值矩阵power和相位矩阵phase成新的矩阵Atheta=[power, phase]
    Atheta = np.concatenate((power, phase), axis=1)

    # 创建Pandas DataFrame
    snapshot_col = ['snapshot' + str(i) for i in range(1, len(IQ) + 1)]
    IQ_df = pd.DataFrame(IQ, columns=['feature' + str(i) for i in range(1, IQ.shape[1] + 1)])
    IQ_df.insert(0, 'time', snapshot_col)

    Atheta_df = pd.DataFrame(Atheta, columns=['feature' + str(i) for i in range(1, Atheta.shape[1] + 1)])
    Atheta_df.insert(0, 'time', snapshot_col)

    # 保存为CSV文件
    IQ_df.to_csv('D:/CodeSpace/PycharmWorkspace/FEDformer-master/dataset/paperDataset/IQ.csv', index=False)
    Atheta_df.to_csv('D:/CodeSpace/PycharmWorkspace/FEDformer-master/dataset/paperDataset/Atheta.csv', index=False)


def loadCFR():
    SNR=0
    # 使用 h5py 从 .mat 文件中加载 CFR 矩阵
    with h5py.File("../dataset/paperDataset/New/data_with_noise/h_CFR_final_SNR{}.mat".format(SNR), 'r') as file:
        # 假设 'h_CFR_final' 是存储复数数据的数据集的名称
        # 你需要根据实际情况调整代码来匹配数据集的结构和名称
        CFR_real = np.array(file['CFR_noise']['real'])  # 加载实部
        CFR_imag = np.array(file['CFR_noise']['imag'])  # 加载虚部
        CFR = CFR_real + 1j * CFR_imag  # 转换为复数数组

        CFR = np.transpose(CFR, axes=(2, 1, 0))  # 可能需要调整维度的顺序
    gateChangeRatio = [0.0915, 0.011, 0.0109, 0.0076]  # [ΔA, Δtheta, ΔI, ΔQ]

    # 对只包含确定性移动的部分，60km/h的运动速度，共10s,1000sample/s，全程运动情况下，
    # IQ的real_part，imag_part在9snapshot内的变化量绝对值的平均值为[0.09148969348668239, 0.011013250188858473]
    # Atheta的A,theta在9snapshot内的变化量绝对值的平均值为[0.010929250449998236, 0.0075608610017994]

    # 找到包含值为"nan"的列
    nan_cols = np.any(np.isnan(CFR), axis=(0, 1))  # 沿着第一和第三维度查找
    # 删除包含值为 "nan" 的列
    CFR = CFR[:, :, ~nan_cols]
    CFR = np.transpose(CFR, axes=(0, 2, 1))  # snapshot * frequency

    # 拆分复数矩阵CIR为实部矩阵realPart和虚部矩阵imagPart
    realPart = np.real(CFR)
    imagPart = np.imag(CFR)

    # 得到连接矩阵IQ=[realPart, imagPart]
    IQ = np.concatenate((realPart, imagPart), axis=2)

    # 计算幅值矩阵power和相位矩阵phase
    power = np.abs(CFR)
    phase = np.angle(CFR)

    # 连接幅值矩阵power和相位矩阵phase成新的矩阵Atheta=[power, phase]
    Atheta = np.concatenate((power, phase), axis=2)

    # 计算C_t
    Ct_IQ = np.zeros_like(IQ)
    for no_antenna in range(Ct_IQ.shape[0]):
        for no_snapshot in range(Ct_IQ.shape[1]):
            Ct_IQ[no_antenna, no_snapshot, :] = calculate_deltas_efficient(IQ[no_antenna, no_snapshot, :])

    Ct_Atheta = np.zeros_like(Atheta)
    for no_antenna in range(Ct_Atheta.shape[0]):
        for no_snapshot in range(Ct_Atheta.shape[1]):
            Ct_Atheta[no_antenna, no_snapshot, :] = calculate_deltas_efficient(Atheta[no_antenna, no_snapshot, :])

    IQ_snapshot = np.zeros([Ct_IQ.shape[0], Ct_IQ.shape[1], 2])
    IQ_antenna = np.zeros([Ct_IQ.shape[1], 6])
    for no_antenna in range(Ct_IQ.shape[0]):
        for no_snapshot in range(Ct_IQ.shape[1]):
            IQ_snapshot[no_antenna, no_snapshot] = [np.sum(Ct_IQ[no_antenna, no_snapshot, 0:511]),
                                                    np.sum(Ct_IQ[no_antenna, no_snapshot, 512:1023])]
            IQ_antenna[no_snapshot, 0] = np.sum(IQ_snapshot[:, no_snapshot, 0])  # 求不同天线的CT的和
            IQ_antenna[no_snapshot, 1] = np.mean(IQ_snapshot[:, no_snapshot, 0])  # 求不同天线的CT的均值
            IQ_antenna[no_snapshot, 2] = np.var(IQ_snapshot[:, no_snapshot, 0])  # 求不同天线的CT的方差
            IQ_antenna[no_snapshot, 3] = np.sum(IQ_snapshot[:, no_snapshot, 1])  # 求不同天线的CT的和
            IQ_antenna[no_snapshot, 4] = np.mean(IQ_snapshot[:, no_snapshot, 1])  # 求不同天线的CT的均值
            IQ_antenna[no_snapshot, 5] = np.var(IQ_snapshot[:, no_snapshot, 1])  # 求不同天线的CT的方差

    Atheta_snapshot = np.zeros([Ct_Atheta.shape[0], Ct_Atheta.shape[1], 2])
    Atheta_antenna = np.zeros([Ct_Atheta.shape[1], 6])
    for no_antenna in range(Ct_Atheta.shape[0]):
        for no_snapshot in range(Ct_Atheta.shape[1]):
            Atheta_snapshot[no_antenna, no_snapshot] = [np.sum(Ct_Atheta[no_antenna, no_snapshot, 0:511]),
                                                        np.sum(Ct_Atheta[no_antenna, no_snapshot, 512:1023])]
            Atheta_antenna[no_snapshot, 0] = np.sum(Atheta_snapshot[:, no_snapshot, 0])  # 求不同天线的CT的和
            Atheta_antenna[no_snapshot, 1] = np.mean(Atheta_snapshot[:, no_snapshot, 0])  # 求不同天线的CT的均值
            Atheta_antenna[no_snapshot, 2] = np.var(Atheta_snapshot[:, no_snapshot, 0])  # 求不同天线的CT的方差
            Atheta_antenna[no_snapshot, 3] = np.sum(Atheta_snapshot[:, no_snapshot, 1])  # 求不同天线的CT的和
            Atheta_antenna[no_snapshot, 4] = np.mean(Atheta_snapshot[:, no_snapshot, 1])  # 求不同天线的CT的均值
            Atheta_antenna[no_snapshot, 5] = np.var(Atheta_snapshot[:, no_snapshot, 1])  # 求不同天线的CT的方差

    # Ct_IQ表示的是原始CFR拆分为实部、虚部时，每个snapshot上实部、虚部的值变化分别达到[{gateChangeRatio[0]},{gateChangeRatio[1]}]时，所需要的snapshot数量(Tc),矩阵维度是[天线数量,snapshot数,(512表示不同频率)512实部+512虚部]
    # IQ_snapshot表示的是Ct_IQ在每根天线，每个snapshot上，对所有频率上的Tc求和，矩阵的维度是[天线数量,snapshot数量,2(实部，虚部)]
    # IQ_antenna表示的是在IQ_snapshot的基础上，对同一snapshot上不同天线全部频率上Tc总和求和、均值、方差，矩阵维度是[snapshot数,6(前三列表示实部部分的和、均值、方差，后三列表虚部部分)]

    # Atheta_IQ表示的是原始CFR拆分为能量、相位时，每个snapshot上能量、相位的值变化分别达到[{gateChangeRatio[2]},{gateChangeRatio[3]}]时，所需要的snapshot数量(Tc),矩阵维度是[天线数量,snapshot数,(512表示不同频率)512实部+512虚部]
    # Atheta_snapshot表示的是Ct_Atheta在每根天线，每个snapshot上，对所有频率上的Tc求和，矩阵的维度是[天线数量,snapshot数量,2(能量，相位)]
    # Atheta_antenna表示的是在Atheta_snapshot的基础上，对同一snapshot上不同天线全部频率上Tc总和求和，均值，方差，矩阵维度是[snapshot数,6(前三列表示能量部分的和、均值、方差，后三列表相位部分)]
    savemat('D:/CodeSpace/PycharmWorkspace/FEDformer-master/dataset/paperDataset/IQ_Ct_MEAN_CFR.mat',
            {'Atheta_antenna': IQ_antenna, 'Atheta_snapshot': IQ_snapshot, "Ct_IQ": Ct_IQ})
    savemat('D:/CodeSpace/PycharmWorkspace/FEDformer-master/dataset/paperDataset/Atheta_Ct_MEAN_CFR.mat',
            {'IQ_antenna': Atheta_antenna, 'IQ_snapshot': Atheta_snapshot, "Ct_IQ": Ct_Atheta})

    # 把不同天线直接拼在一起
    IQ_list = []
    Atheta_list = []
    # 使用循环来遍历每个天线的数据
    for no_antenna in range(IQ.shape[0]):
        # 将对应的切片添加到列表中
        IQ_list.append(IQ[no_antenna, :, :])
        Atheta_list.append(Atheta[no_antenna, :, :])

    # 使用np.stack而不是np.array来堆叠数组，这是一种更高效的方法
    IQ_final = np.stack(IQ_list, axis=0)
    Atheta_final = np.stack(Atheta_list, axis=0)
    # 然后进行transpose和reshape操作
    IQ = IQ_final.transpose(1, 2, 0).reshape(IQ_final.shape[1], -1)
    Atheta = Atheta_final.transpose(1, 2, 0).reshape(Atheta_final.shape[1], -1)

    # 创建Pandas DataFrame
    snapshot_col = ['snapshot' + str(i) for i in range(1, len(IQ) + 1)]
    IQ_df = pd.DataFrame(IQ, columns=['feature' + str(i) for i in range(1, IQ.shape[1] + 1)])
    IQ_df.insert(0, 'time', snapshot_col)

    Atheta_df = pd.DataFrame(Atheta, columns=['feature' + str(i) for i in range(1, Atheta.shape[1] + 1)])
    Atheta_df.insert(0, 'time', snapshot_col)

    # 保存为CSV文件
    IQ_df.to_csv('D:/CodeSpace/PycharmWorkspace/FEDformer-master/dataset/paperDataset/IQ_CFR_SNR{}.csv'.format(SNR), index=False)
    Atheta_df.to_csv('D:/CodeSpace/PycharmWorkspace/FEDformer-master/dataset/paperDataset/Atheta_CFR.csv', index=False)


def loadPDP():
    data = loadmat("../dataset/MaleiSJ/D3_DL_2GHz_3600snaps/CIR_dB.mat")
    pdp = data['CIR_dB']
    # pdp = pdp[0:5000,:]

    interpolation = False
    if (interpolation == True):
        # Load the original data
        # 如果.mat是snapshot*channel则需要转置
        pdp = np.transpose(pdp)
        # Linear interpolation to expand the data

        # 找到包含值为"nan"的列
        nan_cols = np.any(np.isnan(pdp), axis=0)
        # 删除包含值为"nan"的列
        pdp = pdp[:, ~nan_cols]
        ori_num = pdp.shape[1]
        expand_num = 2 * ori_num

        channel_num = pdp.shape[0]  # 通道数
        expanded_pdp = np.zeros((channel_num, expand_num))  # (通道数，snapshot数)

        for i in range(channel_num):
            expanded_pdp[i] = np.interp(np.linspace(0, ori_num, expand_num), np.arange(ori_num), pdp[i])

        # Transpose the data and add the snapshot column
        expanded_pdp = np.transpose(expanded_pdp)
        snapshot_col = np.array([f"snapshot{i + 1}" for i in range(expand_num)]).reshape((expand_num, 1))
        expanded_pdp = np.hstack((snapshot_col, expanded_pdp))

        # Add the time row
        time_row = np.array(["time"] + [f"feature{i + 1}" for i in range(channel_num)]).reshape((1, channel_num + 1))
        expanded_pdp = np.vstack((time_row, expanded_pdp))
    else:
        # Load the original data

        # 如果.mat是snapshot*channel则需要转置
        pdp = np.transpose(pdp)
        # Linear interpolation to expand the data

        # 找到包含值为"nan"的列
        nan_cols = np.any(np.isnan(pdp), axis=0)
        # 删除包含值为"nan"的列
        pdp = pdp[:, ~nan_cols]
        # Transpose the data and add the snapshot column
        expanded_pdp = np.transpose(pdp)
        snapshot_col = np.array([f"snapshot{i + 1}" for i in range(pdp.shape[1])]).reshape((pdp.shape[1], 1))
        expanded_pdp = np.hstack((snapshot_col, expanded_pdp))

        # Add the time row
        time_row = np.array(["time"] + [f"feature{i + 1}" for i in range(pdp.shape[0])]).reshape((1, pdp.shape[0] + 1))
        expanded_pdp = np.vstack((time_row, expanded_pdp))

    # Save the data as a CSV file
    df = pd.DataFrame(expanded_pdp)
    print(df.values.shape)
    df.to_csv("../dataset/combined_delay.csv", index=False, header=False)


def calculate_deltas_efficient(Y_vector, threshold=0.01):
    # 初始化结果向量，与输入向量长度相同，初始值为0
    result = np.zeros(len(Y_vector), dtype=int)

    # 遍历输入向量的每个元素，除了第一个元素，因为它没有前一个元素可以比较
    for t in range(1, len(Y_vector)):
        # 从当前时间点向前查找，直到找到满足条件的点或到达序列开始
        for n in range(1, t + 1):
            # 防止除以零的情况
            if Y_vector[t] == 0:
                continue
            # 计算delta_rows
            delta_rows = abs((Y_vector[t] - Y_vector[t - n]) / Y_vector[t])
            # 检查delta_rows是否大于阈值
            if delta_rows > threshold:
                # 如果是，记录n到结果向量
                result[t] = n
                break  # 找到满足条件的n后不再继续查找

    return result


# loadCIR()
# loadPDP()
loadCFR()
