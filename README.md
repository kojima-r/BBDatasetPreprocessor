# データ準備

Dataset: https://www.kaggle.com/rtatman/british-birdsong-dataset

以下のようにデータを配置
```
BBDatasetPreprocessor/birdsong_metadata.csv
BBDatasetPreprocessor/songs/songs/xc*.flac
```

環境構築
```
conda create -n tf1
conda activate tf1
conda install tensorflow-gpu==1.15
conda install -c conda-forge librosa
pip install joblib
conda install umap
pip install umap-learn
```

DeepKFの中のumap/pcaを使うので次のrepositoryを使用 https://github.com/clinfo/DeepKF
```
pip install --upgrade git+https://github.com/clinfo/DeepKF
```

# 前処理

```
python preprocess.py
python make_dataset.py
```

# 実行＆結果の可視化
```
sh run.sh
```

# 確認のためデータの可視化のみを行う場合
```
python plot.py
```
