# SIGNATE Techno pro

## Requiremnts
下記の環境で動作確認済みです
<pre>
OS: Ubuntu20.04
CPU: Intel 10700K
GPU: NVIDIA RTX3070 8GB
RAM: 64GB

Docker
docker-compose
nvidia-docker
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
        ├── __pycache__
        ├── models
        ├── preprocess.py
        ├── submit.py
        ├── train.py
        └── utils.py
</pre>

## Build
```
docker-compose up
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
./techno/run.sh
```