python main.py test --dataset jzp --config "configs/jzp.yml" --state_dict "./trained_models/wushen_net-v1.ckpt" --gpu 2

python main.py train --dataset jzp --log_dir jzp_logs --exp_name jzp --config "configs/jzp.yml" --gpu 2 --progress_bar

python main.py train_test --dataset jzp --log_dir final_logs --exp_name jzp --config "configs/jzp.yml" --gpu 0 --progress_bar

python main.py kfold --dataset jzp --log_dir kfold_logs --exp_name jzp --config "configs/jzp.yml" --gpu 5 --progress_bar


python main.py test --dataset jzp --config "configs/jzp.yml" --state_dict "./jz_k_ckpt/fold_0/JZ_net_epoch=0.ckpt" --gpu 2

nohup python main.py kfold --dataset jzp --log_dir kfold_logs --exp_name kfold --config "configs/jzp.yml" --gpu 4 --progress_bar > 5fold_150e.out 2>&1 &