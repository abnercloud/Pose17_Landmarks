# Pose17_Landmarks
Pose17_Landmarks人手21关键点模型训练推理。


# 安装

## 环境

此代码是使用带有NVIDIA GPU的Ubuntu 16.04上的Python 3.5和pytorch 0.4.0开发的。使用带有CUDA 9.0和cuDNN 7.0的1个NVIDIA GTX 1080TI GPU进行训练和测试，其他平台或GPU未经过全面测试。

## 依赖项

1. 请按照官方说明安装docker

2. 克隆项目

```
git clone git@github.com:abnercloud/Pose17_Landmarks.git
```

3. 构建docker镜像

```
cd Pose17_Landmarks/
sudo docker build -f ./docker/Dockerfile -t pose17_ldmks:gpu-v0.1 ./docker/
```

4. 启动训练环境

```
sudo docker run --name hand21_ldmks --gpus 0  -it --rm -v $PWD/:/Pose17_Landmarks/ pose17_ldmks:gpu-v0.1 bash
```


# 运行

## 测试

```
cd /Pose17_Landmarks/

python3 demo.py --indir ${img_directory} --outdir ${out_dir} --nClasses 106 --save_img

# example:
python3 demo.py --indir examples/imgs/ --outdir examples/outputs/ --nClasses 106 --save_img
```

## 训练

```
cd /Pose17_Landmarks/train_sppe/src

# Stage 1
python train.py --dataset dt_17_kps --expID stg_1 --nClasses 17 --LR 1e-4 --trainBatch 32 --validBatch 32 --nEpochs 60 --nThreads 30 --inputResH 320 --inputResW 256 --outputResH 80 --outputResW 64 --optMethod adam
# Stage 2
python train.py --dataset dt_17_kps --expID stg_2 --nClasses 17 --LR 1e-5 --trainBatch 32 --validBatch 32 --nEpochs 60 --nThreads 30 --inputResH 320 --inputResW 256 --outputResH 80 --outputResW 64 --optMethod adam --loadModel ../exp/coco/stg_1/model_30.pkl
# Stage 3
python train.py --dataset dt_18_kps --expID stg_2 --nClasses 17 --LR 1e-5 --trainBatch 32 --validBatch 32 --nEpochs 60 --nThreads 30 --inputResH 320 --inputResW 256 --outputResH 80 --outputResW 64 --optMethod adam --loadModel ../exp/coco/stg_2/model_30.pkl --addDPG


## Tensorboard
Stage 1: nohup tensorboard --logdir .tensorboard/coco/stg_1/ &
Stage 2: nohup tensorboard --logdir .tensorboard/coco/stg_2/ --port 6007 &

```

