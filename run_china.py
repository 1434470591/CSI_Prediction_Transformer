import csv
import datetime
import subprocess
import time

# 定义参数组合
SPEED = [
    # 30,
    # 60,
    120,
    'x',
]
length = [
    {'seq_len': 5, 'pred_len': 1, 'slid_step': 1},
    # {'seq_len': 5, 'pred_len': 3, 'slid_step': 1},
    # {'seq_len': 5, 'pred_len': 5, 'slid_step': 1},

]
combinations = [
    {"useEmbedding": 0, "EmbeddingType": 'None', "EmbeddingResponse": "None", "model": "Transformer", "lradj": "cosine", "train_epochs": 256, "learning_rate": 1e-5, "e_layers": 2, "d_layers": 1,},
    {"useEmbedding": 1, "EmbeddingType": "L1", "EmbeddingResponse": "CFR", "model": "Transformer", "lradj": "cosine", "train_epochs": 256, "learning_rate": 1e-5, "e_layers": 2, "d_layers": 1,},
    
    {"useEmbedding": 0, "EmbeddingType": 'None', "EmbeddingResponse": "None", "model": "Informer", "lradj": "cosine", "train_epochs": 256, "learning_rate": 1e-5, "e_layers": 2, "d_layers": 1,},
    {"useEmbedding": 1, "EmbeddingType": "L1", "EmbeddingResponse": "CFR", "model": "Informer", "lradj": "cosine", "train_epochs": 256, "learning_rate": 1e-5, "e_layers": 2, "d_layers": 1,},
]

# 定义种子列表ç
seed_list = [
    2044,
]

# 记录整个实验的开始时间
total_start_time = time.time()

# 迭代并执行每个种子
for seed in seed_list:
    for len in length:
        for speed in SPEED:
            row_data = [[], [datetime.datetime.now().strftime('%Y-%m-%d  *%H:%M:%S'), f"observation windows:{len['seq_len']}", f"prediction windows:{len['pred_len']}", f"speed:{speed}", seed, ]]
            with open("./output_china.csv", mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(row_data)
            # 迭代并执行每个参数组合
            for combo in combinations:
                if combo['model'] == 'Transformer' and speed == 120 :
                    continue
                for axis in [
                    0,
                    1,
                    ]:
                    if combo['useEmbedding'] == 0 and axis != 0:
                        continue
                    for method in [
                                'first', 
                                'second',
                            ]:
                        if combo['useEmbedding'] == 0 and method != 'first':
                            continue
                        for EmbeddingResponse in [
                            'CFR', 
                            'CIR', 
                            'MIX',
                            ]:
                            if combo['useEmbedding'] == 0 and EmbeddingResponse != 'CFR':
                                    continue
                            for EmbeddingType in [
                                'L1', 
                                'L2', 
                                'L3',
                                ]:
                                if combo['useEmbedding'] == 0 and EmbeddingType != 'L1':
                                    continue
                                # 构建命令行参数
                                args = [
                                    "python","main.py",
                                    "--model",combo["model"],
                                    '--useEmbedding', str(combo['useEmbedding']),
                                    '--EmbeddingType', EmbeddingType,
                                    '--EmbeddingResponse', EmbeddingResponse,
                                    '--method', method,
                                    '--axis', str(axis),
                                    '--lradj', combo['lradj'],
                                    "--train_epochs",str(combo["train_epochs"]),
                                    "--learning_rate",str(combo["learning_rate"]),
                                    "--e_layers",str(combo["e_layers"]),
                                    "--d_layers",str(combo["d_layers"]),

                                    "--seq_len",str(len["seq_len"]),
                                    "--label_len",str(len["seq_len"]),
                                    "--pred_len",str(len["pred_len"]),
                                    "--slid_step",str(len["slid_step"]),
                                    "--data",'china',
                                    # single gpu
                                    "--gpu", '3',
                                    # multi gpu
                                    # "--use_multi_gpu", '1',
                                    # "--devices", '2, 3',
                                    '--SNR', 'None',
                                    '--speed', str(speed),
                                    "--seed",str(seed),  # 添加种子参数
                                ]
                                # 记录单轮实验的开始时间
                                start_time = time.time()

                                # 执行脚本
                                print(f"Running with seed {seed}: {' '.join(args)}")
                                subprocess.run(args)

                                # 记录单轮实验的结束时间并计算运行时间
                                end_time = time.time()
                                round_time = end_time - start_time
                                print(f"Round with seed {seed} completed in {round_time:.2f} seconds.\n")

# 记录整个实验的结束时间并计算总运行时间
total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"All rounds completed in {total_time / 60:.2f} minutes.")
