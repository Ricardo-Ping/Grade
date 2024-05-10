import os
from itertools import product

from Model.Grade import Grade
from arg_parser import parse_args, load_yaml_config
from utils import setup_seed, gpu, get_local_time
import torch
import logging
import dataload
from torch.utils.data import DataLoader
from train_and_evaluate import train_and_evaluate

if __name__ == '__main__':
    # 输出参数信息
    args = parse_args()
    # 创建日志文件夹
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置日志文件名
    log_filename = os.path.join(log_dir, f"{args.Model}_{args.data_path}").replace("\\", "/") + ".log"

    # 配置日志格式
    log_format = '%(asctime)s %(levelname)s %(message)s'
    date_format = '%a %d %b %Y %H:%M:%S'

    # 创建一个日志处理器，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)

    # 设置一个文件处理器，用于写入文件
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(console_formatter)

    # 获取 root 记录器并配置处理器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info('============Arguments==============')
    for arg, value in vars(args).items():
        logging.info('%s: %s', arg, value)
    logging.info('local time：%s', get_local_time())
    # 读取设置的不需要修改的参数
    setup_seed(args.seed)
    device = gpu()
    batch_size = args.batch_size
    num_workers = args.num_workers
    dim_E = args.dim_E
    epochs = args.num_epoch
    feature_embedding = args.feature_embed  # 特征嵌入
    model_name = args.Model
    aggr_mode = args.aggr_mode
    # 需要从yaml中读取的参数
    config = load_yaml_config(model_name)
    reg_weight = args.reg_weight
    learning_rate = args.learning_rate
    dropout = args.dropout
    n_layers = args.n_layers
    corDecay = args.corDecay
    n_factors = args.n_factors
    n_iterations = args.n_iterations
    mm_layers = args.mm_layers  # 多模态卷积层数
    ii_topk = args.ii_topk  # 项目-项目图的topk选择
    uu_topk = args.uu_topk  # 用户-用户图的topk选择
    lambda_coeff = args.lambda_coeff  # 跳跃连接系数
    ssl_temp = args.ssl_temp  # ssl的温度系数
    ssl_alpha = args.ssl_alpha  # ssl任务损失的系数
    ae_weight = args.ae_weight  # 自动编码器损失的系数
    threshold = args.threshold  # 去噪门控
    prompt_num = args.prompt_num
    neg_weight = args.neg_weight
    cen_reg = args.cen_reg  # DCCF的意图嵌入正则化
    n_intents = args.n_intents  # DCCF的意图嵌入数量
    G_rate = args.G_rate  # MMSSL的生成器损失权重
    noise_alpha = args.noise_alpha
    ssl_temp2 = args.ssl_temp2

    # 加载训练数据
    train_data, val_data, test_data, user_item_dict, num_user, num_item, v_feat, t_feat = dataload.data_load(
        args.data_path)
    train_dataset = dataload.TrainingDataset(num_user, num_item, user_item_dict, train_data)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    # 网格搜索
    hyper_ls = []
    for param in config['hyper_parameters']:
        hyper_ls.append(config[param])
    # 生成所有可能的超参数组合
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)  # 总组合数量

    # 最佳参数
    best_performance = None
    best_params = None
    best_metrics = None

    for idx, hyper_param_combo in enumerate(combinators):
        hyper_param_dict = dict(zip(config['hyper_parameters'], hyper_param_combo))

        # 输出当前网格搜索序号和总的参数搜索次数
        logging.info('========={}/{}: Parameters:{}========='.format(
            idx + 1, total_loops, hyper_param_dict))

        # 覆盖args中的参数
        for key, value in hyper_param_dict.items():
            setattr(args, key, value)

        # 定义模型
        model_constructors = {
            'LightGCN': lambda: LightGCN(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                         args.n_layers, aggr_mode, device),
            'GSLRec': lambda: GSLRec(num_user, num_item, train_data, user_item_dict, dim_E, args.reg_weight,
                                         args.n_layers, aggr_mode, device),
            'MSFCL': lambda: MSFCL(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E, args.reg_weight,
                                 args.n_layers, args.ssl_temp, args.ssl_alpha, device),
            'MSF': lambda: MSF(num_user, num_item, train_data, user_item_dict, v_feat, t_feat, dim_E,
                                   args.reg_weight,
                                   args.n_layers, args.ssl_temp, args.ssl_alpha, args.ssl_temp2, args.noise_alpha, device),

            # ... 其他模型构造函数 ...
        }
        # 实例化模型
        model = model_constructors.get(model_name, lambda: None)()
        model.to(device)
        for name, param in model.named_parameters():
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            print(f"Parameter requires_grad: {param.requires_grad}")
            # print(f"Parameter data:\n{param.data}")
            print("=" * 30)

        # 定义优化器
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.learning_rate}])

        # 训练和评估
        current_best_metrics = train_and_evaluate(model, train_dataloader, val_data, test_data, optimizer, epochs)

        current_best_recall = current_best_metrics[20]['recall']
        if best_performance is None or current_best_recall > best_performance:
            best_performance = current_best_recall
            best_params = hyper_param_dict.copy()
            best_metrics = current_best_metrics

    # 输出最佳性能和对应的超参数
    logging.info("Best performance: {:.5f}".format(best_performance))
    logging.info("Best parameters: {}".format(best_params))

    # 输出最佳指标
    logging.info("Best metrics:")
    for k, metrics in best_metrics.items():
        metrics_strs = [f"{metric}: {value:.5f}" for metric, value in metrics.items()]
        logging.info(f"{k}: {' | '.join(metrics_strs)}")
