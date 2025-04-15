from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Pred,
    SJTU,
    CHINA,
    SJTU_original,
)

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "sjtu": SJTU,
    "china": CHINA,
    'wbt': SJTU_original,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    if args.data == 'wbt':
        data_set = Data(args, flag)
    else:
        data_set = Data(args=args, flag=flag).__load_data__()

    if flag == "test" :
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == "pred" :
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    print(flag, len(data_set))
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
