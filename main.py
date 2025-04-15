import argparse
import os
import random
import time
import torch
from datetime import datetime
import numpy as np
import pandas as pd


from exp.exp_main import Exp_Main
from utils.tools import getRealNMSE
from test1 import __calculate_delta_CFR_first__, __calculate_delta_CIR_first__, __calculate_deltas_efficient_first__, get_first_tcVec
from test1 import __calculate_delta_CFR_second__, __calculate_delta_CIR_second__, __calculate_deltas_efficient_second__, get_second_tcVec

import os

def main():
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    parser.add_argument('--seed', type=int, default=2044, help='random seed')
    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='Transformer',
                        help='model name, options: [FEDformer,Autoformer,Informer,Transformer, DLinear, SCINet, ConvFC, MTSMixer, MTSMatrix, FNet,LSTM,TransformerRPE]')

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='softmax',
                        help='mwt cross atention activation function tanh or softmax')
    
    # data loader
    parser.add_argument('--root_path', type=str, default='./dataset/paperDataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='IQ_CFR_SNR300.csv', help='data file')
    parser.add_argument("--data", type=str, default='china', help='dataset type, options:[china(mobile communication open dataset), sjtu(data services lab dataset)]')
    parser.add_argument('--groundtruth_path', type=str, default='UMA4Rx32Tx5Ms8RB30km.npy', help='data file')
    # parser.add_argument('--groundtruth_path', type=str, default='IQ_CFR_nonoise.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='feature1', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='s',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument("--save_name", type=str, default="/checkpoint.pth")
    parser.add_argument('--test_len', type=int, default=0, help='length of test data')

    # data prepocess
    parser.add_argument("--enhancement", type=int, default=0, help="options:[0, 1]")
    parser.add_argument('--SNR', type=str, default=0)
    parser.add_argument("--speed", type=str, default=30, help="data enviroment")
    parser.add_argument("--scenario", type=str, default='SISO', help="options:[MIMO, SISO]")
    parser.add_argument("--space", type=int, default=0, help="whether to use space model")
    parser.add_argument("--split_ratio", type=list, default=[0.8, 0.1, 0.1], help="percentage of train data",)
    parser.add_argument("--scale", type=bool, default=True, help='standardization')
    parser.add_argument("--data_mean", type=float, default=0, help="mean of array")
    parser.add_argument("--data_std", type=float, default=1.0, help="std of array")
    parser.add_argument("--Txth", type=int, default=1, help="Location of tx Tx")
    parser.add_argument("--Rxth", type=int, default=1, help="Location of tx Rx")
    parser.add_argument("--subcarrier_num", type=int, default=32, help="number of subcarriers")
    parser.add_argument("--RB_num", type=int, default=8, help="number of Resource Block")

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=32, help='input sequence length')
    parser.add_argument("--slid_step", type=int, default=1, help="window shift step")
    parser.add_argument('--label_len', type=int, default=32, help='start token length')
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--seg', type=int, default=20, help='prediction plot segments')
    parser.add_argument('--rev', action='store_true', default=False, help='whether to apply RevIN')
    parser.add_argument('--norm', action='store_false', default=True, help='whether to apply LayerNorm')
    parser.add_argument('--fac_T', action='store_true', default=False,
                        help='whether to apply factorized temporal interaction')
    parser.add_argument('--sampling', type=int, default=2,
                        help='the number of downsampling in factorized temporal interaction')
    parser.add_argument('--fac_C', action='store_true', default=False,
                        help='whether to apply factorized channel interaction')
    parser.add_argument('--refine', action='store_true', default=False, help='whether to refine the linear prediction')
    parser.add_argument('--mat', type=int, default=0, help='option: [0-random, 1-identity]')

    # model define
    parser.add_argument('--num_layers', type=int, default=4, help='LSTM layers')
    parser.add_argument('--embed_type', type=int, default=2,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + positional embedding')
    parser.add_argument('--enc_in', type=int, default=64, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=64, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=64, help='output size')
    parser.add_argument('--d_model', type=int, default=2048, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='learned',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', default=True, help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=128, help='train epochs')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warm up epochs')
    parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
    parser.add_argument('--delta', type=float, default=1e-6, help='')
    parser.add_argument('--learning_rate', type=float, default=0.000001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='consine', help='options=[cosine, original, setLR]')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu number')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='2, 3', help='device ids of multi gpus')
    
    # Embedding
    parser.add_argument('--useEmbedding', type=int, default=0)
    parser.add_argument('--tcVec', type=int, nargs='+', default=[])
    parser.add_argument('--EmbeddingType', type=str, default='L1', help='options:[None, L1, L2,...,Lp]')
    parser.add_argument('--EmbeddingResponse', type=str, default='CFR', help='options:[CFR, CIR]')
    parser.add_argument('--method', type=str, default='ds', help='options:[first, second]')
    parser.add_argument('--axis', type=int, default=1, help='options:[0(kai ying tc embedding), 1(normal tc embedding)]')
    parser.add_argument('--threshold', type=float, default=0.5, help='Embedding threshold')

    args = parser.parse_args()
    SNR = args.SNR
    speed = args.speed

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)# CPU
    np.random.seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)# GPU
    os.environ["PYTHONHASHSEED"] = str(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    if args.useEmbedding == 0:
        args.EmbeddingType = args.EmbeddingResponse = args.method = 'None'
    
    # select dataset
    if args.data == 'sjtu':
        args.root_path = './dataset/'
        args.data_path = f"IQ_CFR_SNR{args.SNR}.csv"
        args.batch_size = 64
        args.subcarrier_num = 32 * 2 if args.enhancement == 1 else 32
        # Embedding
        if args.useEmbedding and args.axis:
            if args.method == 'second':
                # args.threshold = 0.7
                CFR_vector, CIR_vector = __calculate_delta_CFR_second__(path=(args.root_path + args.data_path), axis=args.axis), __calculate_delta_CIR_second__(path=args.root_path + args.data_path, axis=args.axis)
                args.tcVec =(__calculate_deltas_efficient_second__(CFR_vector, EmbeddingType=args.EmbeddingType, threshold=args.threshold) + __calculate_deltas_efficient_second__(CIR_vector, EmbeddingType=args.EmbeddingType, threshold=args.threshold)) / 2 if args.EmbeddingResponse == 'MIX' else __calculate_deltas_efficient_second__(Y_vector=CFR_vector if args.EmbeddingResponse == 'CFR' else CIR_vector, EmbeddingType=args.EmbeddingType, threshold=args.threshold) 
            elif args.method == 'first':
                # args.threshold = 0.07
                CFR_vector_L1, CFR_vector_L2 = __calculate_delta_CFR_first__(path="./dataset/paperDataset/New/data_with_noise/h_CFR_final_SNR{}.mat".format(SNR), axis=args.axis)
                CIR_vector_L1, CIR_vector_L2 = __calculate_delta_CIR_first__(path="./dataset/paperDataset/New/data_with_noise/h_CFR_final_SNR{}.mat".format(SNR), axis=args.axis)
                if args.EmbeddingResponse == 'MIX' :
                    L1vector, L2vector =(__calculate_deltas_efficient_first__(CFR_vector_L1, args.threshold) + __calculate_deltas_efficient_first__(CIR_vector_L1, args.threshold)) / 2, (__calculate_deltas_efficient_first__(CFR_vector_L2, args.threshold) + __calculate_deltas_efficient_first__(CIR_vector_L2, args.threshold)) / 2 
                else :
                    L1vector, L2vector = __calculate_deltas_efficient_first__(CFR_vector_L1 if args.EmbeddingResponse == 'CFR' else CIR_vector_L1, args.threshold), __calculate_deltas_efficient_first__(CFR_vector_L2 if args.EmbeddingResponse == 'CFR' else CIR_vector_L2, args.threshold)
                args.tcVec = L1vector if args.EmbeddingType == 'L1' else L2vector
    
    elif args.data == 'china': 
        args.root_path = './dataset/china/'
        args.data_path = f"UMA4Rx32Tx5Ms8RB{args.speed}km.npy"
        args.batch_size = 1024
        args.subcarrier_num = args.RB_num
        if args.scenario == 'SISO':
            # Embedding
            if args.useEmbedding and args.axis:
                if args.method == 'second' :
                    args.tcVec = get_second_tcVec(f"./dataset/china/UMA4Rx32Tx5Ms8RB{args.speed}km.npy", args.EmbeddingType, args.EmbeddingResponse, args.threshold, args.scenario, args.Txth, args.Rxth)
                elif args.method == 'first' :
                    args.tcVec = get_first_tcVec(f"./dataset/china/UMA4Rx32Tx5Ms8RB{args.speed}km.npy", args.EmbeddingType, args.EmbeddingResponse, 0.24, args.scenario, args.Txth, args.Rxth)
        elif args.scenario == 'MIMO':
            pass
    args.enc_in = args.dec_in = args.c_out = args.subcarrier_num * 2

    print('Args in experiment:')
    if args.is_training:
        # Namespace
        print(args)
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_seed{}'.format(
                args.task_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii,
                args.useEmbedding,
                args.EmbeddingType, 
                args.EmbeddingResponse, 
                args.method, 
                args.axis,
                speed,
                SNR,
                args.enhancement,
                fix_seed
                )
            
            exp = Exp_Main(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, load=0)

            torch.cuda.empty_cache()
    else:
        # Namespace
        print(args)
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_seed{}'.format(
                args.task_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii,
                args.useEmbedding,
                args.EmbeddingType, 
                args.EmbeddingResponse, 
                args.method, 
                args.axis,
                speed,
                SNR,
                args.enhancement,
                fix_seed
                )

            exp = Exp_Main(args)  # set experiments

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, load=1)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting)
                
            torch.cuda.empty_cache()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    test = np.load(f'results/{setting}/metrics.npy', encoding="latin1")  # 加载文件
    if args.data == 'sjtu':
        realMSE, realNMSE = getRealNMSE(f'results/{setting}/', SNR, 0)
    else :
        realMSE, realNMSE = getRealNMSE(f'results/{setting}/', speed, 1)
    nmse_dB = 10 * np.log10(realNMSE)
    reals = "反归一化后：\n MSE:{} \n NMSE:{} \nNMSE(dB):{}".format(realMSE, realNMSE, nmse_dB)

    LOSS = " MAE:{} \n MSE:{} \n RMSE:{} \n MAPE:{} \n MSPE:{} \n NMSE:{} \n SGCS:{} \n FLOPs:{} \n Params:{}".format(test[0],
                                                                                                                    test[1],
                                                                                                                    test[2],
                                                                                                                    test[3],
                                                                                                                    test[4],
                                                                                                                    test[5],
                                                                                                                    test[6],
                                                                                                                    test[7],
                                                                                                                    test[8])
    # 定义要写入的数据，这里是一个包含"A"和"B"的列表
    data_to_append = pd.DataFrame(
        [[current_time, args.is_training, args.model,  reals, LOSS, SNR, test[6] / 1000 ** 3,fix_seed, args]],
        columns=['Time', 'is_training', 'Model', 'FinalLoss', 'loss', 'SNR', 'FLOPs/G', 'fix_seed','args']
        )

    # 定义Excel文件的路径
    excel_path = 'dataset/exp_record' + str(fix_seed) + '.csv'

    # 检查文件是否存在
    if not os.path.exists(excel_path):
        # 如果文件不存在，创建一个新的DataFrame并保存为Excel文件
        data_to_append.to_csv(excel_path, index=False, encoding='utf-8')
    else:
        # 如果文件已存在，读取Excel文件
        existing_data = pd.read_csv(excel_path, encoding='utf-8')

        # 将新数据追加到现有数据上
        updated_data = existing_data._append(data_to_append, ignore_index=True)

        # 将更新后的数据保存回Excel文件
        updated_data.to_csv(excel_path, index=False, encoding='utf-8')

    print(f"Data appended to {excel_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    # 计算总运行时间
    total_time = end_time - start_time

    # 输出总运行时间
    print(f"程序总运行时间：{total_time}秒")
