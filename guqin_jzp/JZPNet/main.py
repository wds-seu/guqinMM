import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from datamodules import Vocabulary, JZPDataModule
from models import tad_module


def get_args_parser():

    parser = argparse.ArgumentParser(description="TADNet")
    subparsers = parser.add_subparsers()
    jzp_parser = subparsers.add_parser('train_test')
    jzp_parser.set_defaults(func=train_test)
    jzp_parser.add_argument("--log_dir", default='logs', help="TensorBoard log output directory")
    jzp_parser.add_argument("--exp_name", default='jzp')
    jzp_parser.add_argument("--seed", default=42, type=int)
    jzp_parser.add_argument("--gpu", default=0, type=int)
    jzp_parser.add_argument("--progress_bar", action="store_true")
    jzp_parser.add_argument(
        "--config",
        default=None,
        help="path to model config file, see \"configs/crohme.yml\" for example"
    )
    jzp_parser.add_argument(
        "--state_dict",
        default=None,
        help="path to the state_dict of the model to be tested")
    
    jzp_parser.add_argument("--dataset", default='jzp')

    kfold_parser = subparsers.add_parser('kfold')
    kfold_parser.set_defaults(func=kfold)
    kfold_parser.add_argument("--log_dir", default='logs', help="TensorBoard log output directory")
    kfold_parser.add_argument("--exp_name", default='jzp')
    kfold_parser.add_argument("--seed", default=42, type=int)
    kfold_parser.add_argument("--gpu", default=0, type=int)
    kfold_parser.add_argument("--progress_bar", action="store_true")
    kfold_parser.add_argument(
        "--config",
        default=None,
        help="path to model config file, see \"configs/crohme.yml\" for example"
    )
    kfold_parser.add_argument(
        "--state_dict",
        default=None,
        help="path to the state_dict of the model to be tested")
    kfold_parser.add_argument("--dataset", default='jzp')

    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)
    test_parser.add_argument(
        "--config",
        default=None,
        help="path to model config file, see \"configs/crohme.yml\" for example"
    )
    test_parser.add_argument(
        "--state_dict",
        default=None,
        help="path to the state_dict of the model to be tested")
    test_parser.add_argument("--gpu", default=0, type=int)

    return parser


def train_test(args):
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(args.seed)
    node_vocab = Vocabulary("data/jzp/node_dict.txt")
    edge_vocab = Vocabulary("data/jzp/edge_dict.txt",
                            use_sos=False,
                            use_eos=False)
    dm = JZPDataModule(node_vocab, edge_vocab)
    dm.setup()
    logger = TensorBoardLogger(args.log_dir, args.exp_name)
    if not args.config:
        if args.dataset == 'jzp':
            config_path = "configs/jzp.yml"
    else:
        config_path = args.config
    config_f = open(config_path, 'r')
    config_dict = yaml.load(config_f, Loader=yaml.FullLoader)
    config_f.close()
    model = tad_module.TADNet(**config_dict)
    checkpoint_callback = ModelCheckpoint(monitor="val_ExpRate",
                                          save_top_k=2,
                                          mode="max",
                                          save_last=True,
                                          dirpath="jz_ckpt/",
                                          filename="JZ_net")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, lr_monitor]
    trainer = pl.Trainer(gpus=[args.gpu],
                         callbacks=callbacks,
                         logger=logger,
                         enable_progress_bar=args.progress_bar,
                         max_epochs=1,
                         gradient_clip_val=1)
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    #TEST STAGE
    model.eval()
    with torch.no_grad():
        print("Testing on JZP:")
        model.beam_test(dm.test_dataloader(), node_vocab, edge_vocab)

def test(args):
    node_vocab = Vocabulary("data/jzp/zyt/node_dict.txt")
    edge_vocab = Vocabulary("data/jzp/zyt/edge_dict.txt",
                            use_sos=False,
                            use_eos=False)
    dm = JZPDataModule(node_vocab, edge_vocab)
    dm.setup("test")
    if not args.config:
        config_path = "configs/jzp.yml"
    else:
        config_path = args.config

    config_f = open(config_path, 'r')
    config_dict = yaml.load(config_f, Loader=yaml.FullLoader)
    config_f.close()
    model = tad_module.TADNet(**config_dict)
    state_dict_path = args.state_dict
    checkpoint = torch.load(state_dict_path,
                            torch.device('cuda', args.gpu))

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        print("Testing on JZP:")
        model.beam_test(dm.test_dataloader(), node_vocab, edge_vocab)
       # model.predict_result(dm.test_dataloader(), node_vocab, edge_vocab)

def kfold(args):
    global best_val_score
    global best_model_path

    # 定义折数
    n_split = 5
    # 初始化数据模块，传入当前折数
    best_val_score = float('-inf')
    best_model_path = None
    node_vocab = Vocabulary("data/jzp/node_dict.txt")
    edge_vocab = Vocabulary("data/jzp/edge_dict.txt",
                            use_sos=False,
                            use_eos=False)
    dm = JZPDataModule(
        node_vocab=node_vocab, 
        edge_vocab=edge_vocab, 
        n_splits=n_split, 
    )

    
    for fold in range(n_split):

        print(f"Start {fold + 1}/{n_split}th train")
        dm.current_fold = fold
        dm.setup()
        logger = TensorBoardLogger(
                save_dir=args.log_dir, 
                name=f"{args.exp_name}_fold_{fold}"
            )
        
        if not args.config:
            if args.dataset == 'jzp':
                config_path = "configs/jzp.yml"
        else:
            config_path = args.config
        config_f = open(config_path, 'r')
        config_dict = yaml.load(config_f, Loader=yaml.FullLoader)
        config_f.close()

        model = tad_module.TADNet(**config_dict)
        checkpoint_callback = ModelCheckpoint(
                monitor="val_ExpRate",
                save_top_k=2,
                mode="max",
                save_last=True,
                dirpath=f"jz_k_ckpt/fold_{fold}/",
                filename="JZ_net_{epoch}",
            )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks = [checkpoint_callback, lr_monitor]
        trainer = pl.Trainer(gpus=[args.gpu],
                                callbacks=callbacks,
                                logger=logger,
                                enable_progress_bar=args.progress_bar,
                                max_epochs=150,
                                gradient_clip_val=1)
        trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
        
        val_result = trainer.validate(model, datamodule=dm)
        val_score = val_result[0]['val_ExpRate']

        # 检查是否是最佳模型
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_path = checkpoint_callback.best_model_path
        print(f"Fold {fold + 1}/{n_split}th train finished")
        print("bset model path: ", best_model_path)
        print("best val score: ", best_val_score)
        # 切换到下一个折
        dm.next_fold()

    print("在测试集上测试最终模型")
    z_node_vocab = Vocabulary("data/jzp/zyt/node_dict.txt")
    z_edge_vocab = Vocabulary("data/jzp/zyt/edge_dict.txt",
                            use_sos=False,
                            use_eos=False)
    dm_zyt = JZPDataModule(z_node_vocab, z_edge_vocab)
    dm_zyt.setup("test")
    best_cp = torch.load(best_model_path,
                            torch.device('cuda', args.gpu))
    model.load_state_dict(best_cp['state_dict'])
    model.eval()
    with torch.no_grad():
        model.beam_test(dm.test_dataloader(), node_vocab, edge_vocab)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.func(args)
