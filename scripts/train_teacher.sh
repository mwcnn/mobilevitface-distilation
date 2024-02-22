# Teacher Network (HR) -> down_size 0 denotes the HR settings
python train_teacher.py --gpus 0 --save_dir checkpoint/teacher/ --down_size 128 --batch_size 128 --data_dir /data/sung/dataset/Face

# LR network w/o distillation (baseline)
python train_teacher.py --gpus 0 --save_dir checkpoint/base_14/ --down_size 16 --batch_size 128 --data_dir /data/sung/dataset/Face
python train_teacher.py --gpus 0 --save_dir checkpoint/base_28/ --down_size 32 --batch_size 128 --data_dir /data/sung/dataset/Face
python train_teacher.py --gpus 0 --save_dir checkpoint/base_56/ --down_size 64 --batch_size 128 --data_dir /data/sung/dataset/Face