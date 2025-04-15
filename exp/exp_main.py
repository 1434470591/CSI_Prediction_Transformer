import csv
import logging
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from matplotlib import pyplot as plt
from thop import profile
from torch import optim


from data_provider.data_factory import data_provider
from data_provider.data_groundtruth_factory import data_provider_gt
from exp.exp_basic import Exp_Basic
from models import (\
    Transformer,
    Informer,
    FEDformer,
    Autoformer,
    DLinear,
    SCINet,
    MTSMixer,
    MTSMatrix,
    MTSAttn,
    FNet,
    MTSD,
    TransformerRPE,
)
from utils.metrics import metric, get_vector_modulus
from utils.tools import EarlyStopping, visual, adjust_learning_rate1, adjust_learning_rate2

warnings.filterwarnings("ignore")

non_transformer = [
    "DLinear",
    "SCINet",
    "MTSMixer",
    "MTSMatrix",
    "MTSAttn",
    "FNet",
    "Transformer_lite",
    "MTSD",
]
logging.getLogger("thop").setLevel(logging.WARNING)


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            "FEDformer": FEDformer,
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "Informer": Informer,
            "DLinear": DLinear,
            "SCINet": SCINet,
            "MTSMixer": MTSMixer,
            "MTSMatrix": MTSMatrix,
            "MTSAttn": MTSAttn,
            "FNet": FNet,
            "MTSD": MTSD,
            "TransformerRPE": TransformerRPE,
        }
        a = self.args
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            # model = nn.parallel.DistributedDataParallel(model, device_ids = self.args.device_ids)
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameters: %.4fM" % (total / 1024 ** 2))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_gt_data(self, flag):
        data_set, data_loader = data_provider_gt(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # for j, (x, y, x_mark, y_mark) in enumerate(gt_vali_loader):
                #     if j == i:
                #         batch_y = y.float().to(self.device)
                #         batch_y_mark = y_mark.float().to(self.device)
                #         break

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device))
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in non_transformer:
                            outputs = self.model(batch_x)
                        else:
                            outputs = (
                                self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                if self.args.output_attention
                                else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            )
                else:
                    if self.args.model in non_transformer:
                        outputs = self.model(batch_x)
                    else:
                        outputs = (
                            self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            if self.args.output_attention
                            else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        )

                f_dim = -1 if self.args.features == "MS" else 0
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        plt.clf()
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="vali")
        test_data, test_loader = self._get_data(flag="test")
        # gt_train_data, gt_train_loader = self._get_gt_data(flag="groundtruth_train")
        # gt_vali_data, gt_vali_loader = self._get_gt_data(flag="groundtruth_vali")
        # gt_test_data, gt_test_loader = self._get_gt_data(flag="groundtruth_test")

        visual_train_loss = []
        visual_vali_loss = []
        visual_test_loss = []

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        print('train_step:', train_steps)
        warmup_steps = train_steps * self.args.warmup_epochs
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=self.args.delta)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.scenario == 'china':
            self.args.use_amp = True
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # for j, (x, y, x_mark, y_mark) in enumerate(gt_train_loader):
                #     if j == i:
                #         batch_y = y.float().to(self.device)
                #         batch_y_mark = y_mark.float().to(self.device)
                #         break

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device))

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in non_transformer:
                            outputs = self.model(batch_x)
                        else:
                            outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                if self.args.output_attention
                                else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            )

                        f_dim = -1 if self.args.features == "MS" else 0
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model in non_transformer:
                        outputs = self.model(batch_x)
                    else:
                        outputs = (
                            self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            if self.args.output_attention
                            else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        )

                    f_dim = -1 if self.args.features == "MS" else 0
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\t iters: {0} | loss: {1:.7f}".format(i + 1, loss.item()))

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                adjust_learning_rate2(model_optim, i + 1 + epoch * train_steps, warmup_steps, self.args)

            print("\tEpoch: {} cost time: {} seconds".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            visual_train_loss.append(train_loss)
            visual_test_loss.append(test_loss)
            visual_vali_loss.append(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(epoch, vali_loss, self.model, path)
            if early_stopping.early_stop == True:
                print("Early stopping")
                break

            # adjust_learning_rate1(model_optim, epoch + 1, self.args)

        # save jpg
        plt.plot(visual_train_loss, label="Train Loss")
        # plt.plot(visual_test_loss, label="Test Loss")
        plt.plot(visual_vali_loss, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        if not os.path.exists(r"./lossJPG"):
            os.makedirs(r"./lossJPG")
        plt.savefig(r"./lossJPG/{}".format(self.args.model))

        return self.model

    def test(self, setting, load=0):
        plt.cla()
        test_data, test_loader = self._get_data(flag="test")
        # gt_test_data, gt_test_loader = self._get_gt_data(flag="groundtruth_test")
        # load model
        if load:
            print("loading model")
            
            best_model_path = self.args.checkpoints + setting + self.args.save_name
            print(best_model_path)
            best_checkpoint = torch.load(best_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=False)

            self.model.load_state_dict(best_checkpoint['model_state_dict'])

        flops = 0
        params = 0
        preds = []
        trues = []
        inputx = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # for j, (x, y, x_mark, y_mark) in enumerate(gt_test_loader):
                #     if j == i:
                #         batch_y = y.float().to(self.device)
                #         batch_y_mark = y_mark.float().to(self.device)
                #         break

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device))
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model in non_transformer:
                            outputs = self.model(batch_x)
                            flops1, params1 = profile(self.model, inputs=(batch_x,))
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[
                                0]if self.args.output_attention else \
                                self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                            flops1, params1 = profile(self.model,inputs=(batch_x,batch_x_mark,dec_inp,batch_y_mark,),)
                else:
                    if self.args.model in non_transformer:
                        outputs = self.model(batch_x)
                        flops1, params1 = profile(self.model, inputs=(batch_x,))
                    else:
                        outputs = (
                            self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            if self.args.output_attention
                            else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        )
                        flops1, params1 = profile(self.model,inputs=(
                                batch_x,
                                batch_x_mark,
                                dec_inp,
                                batch_y_mark,
                            ),
                        )

                f_dim = -1 if self.args.features == "MS" else 0

                flops += flops1
                params += params1

                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                err = np.abs((true - pred) / true) * 100
                if np.all(err <= 20):  # 是否预测值每一位和真实值误差都在20%以内
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.array(preds)
        trues = np.array(trues)
        # inputx = np.array(inputx)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        print("test shape transform to :", preds.shape, trues.shape)
        
        # inverse
        preds = preds * self.args.data_std + self.args.data_mean
        trues = trues * self.args.data_std + self.args.data_mean
        
        # metrics
        mae, mse, rmse, mape, mspe, nmse, nmse_db, SGCS = metric(preds[:, :, :64], trues[:, :, :64])
        # flops = flops / (len(test_loader) * self.args.batch_size)
        flops = flops / (len(test_loader) * self.args.batch_size)
        params = params / (len(test_loader))
        TOPS = flops / (5e-3 * 1000 ** 4)
        # log
        print(f"nmse{nmse}, nmsedb{nmse_db}, SGCS: {SGCS}, FLOPs: {flops / 1000 ** 3}G, Params: {params / 1000 ** 2}M")
        # result save npy
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe, nmse, SGCS, flops, params]),)
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        # txt
        f = open("result.txt", "a")
        f.write(setting + "\n")
        f.write("mae:{}, rmse:{}, nmse{}, SGCS:{}, FLOPs:{}M, Params:{}K".format(
            mae, rmse, nmse, SGCS, flops / 1000**2, params / 1000**1))
        f.write("\n")
        f.write("\n")
        f.close()

        # csv
        row_data = [[self.args.model, self.args.useEmbedding, f'axis: {self.args.axis}', self.args.method, self.args.EmbeddingResponse, self.args.EmbeddingType, self.args.threshold, round(mse, 6), round(nmse, 6), round(nmse_db, 6), round(SGCS, 6), f'flops:{round(flops / 1000**2, 4)}M', f'params:{round(params / 1024**2, 4)}M', f'TOPS:{round(TOPS, 8)}']]
        with open(f"./output_{self.args.data}.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(row_data)
        
        self.visualize(setting)

        return
    
    def visualize(self, setting):
        plt.clf()
        source_path = "./results/" + setting + "/"
        
        pred_path = os.path.join(source_path, "pred.npy")
        true_path = os.path.join(source_path, "true.npy")
        pred = np.load(pred_path, allow_pickle=True)
        true = np.load(true_path, allow_pickle=True)

        start_snapshot = 0
        end_snapshot = 100
        timestamp = 0
        feature_index = 0
        print("shape of pred:", pred.shape)
        print("shape of true:", true.shape)
        pred = pred[start_snapshot:end_snapshot, timestamp, feature_index]
        true = true[start_snapshot:end_snapshot, timestamp, feature_index]

        plt.figure(figsize=(20, 10))
        plt.plot(pred, marker='o', markeredgecolor="black", linestyle="--", label='Pred', color = 'r',)
        plt.plot(true, color='b', label='True')
        plt.xlabel('Row Number')
        plt.ylabel('Value')
        plt.legend()
        if self.args.scenario == 'china':
            plt.title(r'model{}_speed{}'.format(self.args.model, self.args.speed))
        else: 
            plt.title(r'model{}_snr{}'.format(self.args.model, self.args.SNR))
        plt.show()
        plt.savefig('{}visualize.png'.format(source_path))

        return 
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            print('load model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting + self.args.save_name), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []
        inputx = []
        flops = params = 0
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(pred_loader))
        with torch.no_grad():
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
            dec_inp = (torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device))
            # encoder - decoder
            inference_start = time.time()

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.model in non_transformer:
                        outputs = self.model(batch_x)
                        inference_time = time.time() - inference_start
                        flops, params = profile(self.model, inputs=(batch_x,))
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]if self.args.output_attention else \
                            self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        inference_time = time.time() - inference_start
                        
                        flops, params = profile(self.model,inputs=(batch_x,batch_x_mark,dec_inp,batch_y_mark,),)
            else:
                if self.args.model in non_transformer:
                    outputs = self.model(batch_x)
                    inference_time = time.time() - inference_start
                    flops, params = profile(self.model, inputs=(batch_x,))
                else:
                    outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]if self.args.output_attention
                        else self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
                    inference_time = time.time() - inference_start
                    
                    flops, params = profile(self.model,inputs=(batch_x, batch_x_mark, dec_inp, batch_y_mark,),)
            
            f_dim = -1 if self.args.features == "MS" else 0

            batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
        
        print(batch_x.shape)
        TOPS = (flops / 5e-3 )/ 1e12
        print(flops, params, inference_time, TOPS)

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return
