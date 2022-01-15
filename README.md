# SIGNATE TECHNOPRO 2nd place solution

- SIGNATE TECHNOPROコンペの最終submissionコードです。
- 解法自体は[こちら](https://docs.google.com/presentation/d/1BYuW5Qwg916QPBuMg0fH6RaV_pFs9Yw32osqBBzNvfs/edit?usp=sharing)に記載しています。

## Enviroment
下記の環境で動作確認済みです
<pre>
OS: Ubuntu20.04 LTS 
CPU: Intel Corei7 10700k
GPU: NVIDIA RTX3070 8GB
RAM: 64GB

CUDA: 11.3
Docker: 20.10.9
docker-compose: 1.29.2
nvidia-docker: 2.8.0
</pre>

## Directory Tree
<pre>
.
├── README.md
├── docker-compose.yml
├── Dockerfile
└── techno
    ├── __init__.py
    ├── config
    │   ├── preprocess_cfg.yaml
    │   └── train_cfg.yaml
    ├── data
    │   └── raw_data
    ├── output
    ├── run.sh
    └── src
        ├── models
        ├── preprocess.py
        ├── submit.py
        ├── train.py
        └── utils.py
</pre>

## Build
```
git clone https://github.com/tubo213/signate_technopro.git
cd signate_technopro
docker-compose up -d
docker exec -it signate_technopro_competition_1 /bin/bash
```

## How to Run

### Data download
コンペのデータをダウンロードし，./techno/data/raw_dataに解凍してください．


### Preprocess
GaussianMixtureによる異常検知でテストデータに疑似ラベルを貼ります．
```
python ./techno/src/preprocess.py
```

別のyamlファイルを指定して実行することも可能です．  
デフォルトでは./techno/config/preprocess_cfg.yamlを参照します．
```
python ./techno/src/preprocess.py -c hoge.yaml
```

### Training
疑似ラベリングと学習を繰り返し行います．
```
python ./techno/src/train.py
```
こちらも別のyamlファイルを指定して実行することが可能です．  
デフォルトでは./techno/config/train_cfg.yamlを参照します．

```
python ./techno/src/train.py -c hoge.yaml
```

### Make submission file
学習済みモデルのディレクトリを指定してsubmissionファイルの作成を行います．
```
python ./techno/src/submit.py -d ../output/exp_001/swin_tiny_patch4_window7_224/thresh85/default/
```

### Make final submission
最終submission fileは下記コマンドで作成できます．

```
bash ./techno/run.sh
```

## References
- https://signate.jp/courses/OYGzkg6XZYWldZ3N/leaderboard
- Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, James Bailey Symmetric Cross Entropy for Robust Learning with Noisy Labels https://arxiv.org/abs/1908.06112 16 Aug 2019
- https://www.kaggle.com/phalanx/train-swin-t-pytorch-lightning
