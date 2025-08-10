# TASTE
<div align="center">
    <img src="pics/logo.png" alt="logo" width="400">
</div>



TASTE is a framework for content-augumented Music Recommendation. It is developed for reproducing and developing recommendation algorithms with content more efficiently.

The proposed subset of lfm-1b and corresponding embedded wav features can be downloaded here: [Google Drive](https://drive.google.com/drive/folders/1H-wrqchl-QMWrO-13mueeO5t-7nL00JU?usp=sharing)

## Overall

<div align="center">
    <img src="pics/framework.png" alt="framework" width="800">
    <p>Figure : Overview of the TASTE framework for content-augmented music recommendation.</p>
</div>

This framework includes feature extraction, feature fusion, model training, and testing. 

The traditional features are embedded in the way commonly followed: discrete features are processed using one-hot encoding, then mapped to lower-dimensional continuous vectors; continuous features are discretized according to the defined method by default, and then treated as discrete features (other methods can also be used to handle continuous features).

After the audio and text features are extracted, they pass through trainable MLP layers and are then merged with the previously extracted features to be input into various models, which ultimately output the predicted click-through rate.

Furthermore, we applied the **MuQ-token** method to the multi-layer output of our ""MuQ"" model. By using discrete tokens to represent the audio features, we were able to achieve better results

Our method is highly compatible with various models because it only adds more features without requiring any changes to the model's structure itself.

## Usage

Clone this repository to the local machine, then install the dependencies.
```
pip install -r requirements.txt
```

If you want to manually extract audio features based on your own data and model, you can use the `./notebook/extract_feature.ipynb` in the `./notebooks/` directory. For detailed instructions, please refer to the script files.

### Quick-Start

After downloading the data, place it in `datas\lfm1b-filtered` and then run the following command:

```
python main.py
```

This command runs the framework with the default settings in the simplest way. If adjustments are needed afterward, parameters can be set either through the command line or by using YAML files，for example:

```
python main.py --model_name LR --dataset_name lfm1n-filtered ----config_files config/config1.yaml
```


## Models

Currently, we have implemented the following models on TASTE:

### CTR Task
#### Context-aware Models

| Model | Publish | paper name |
| :--- | :--- | :--- |
| LR | WWW '07 | Predicting clicks: estimating the click-through rate for new ads |
| FM | ICDM'10     | Factorization Machines      
| FFM | RecSys '16 | Field-aware factorization machines for CTR prediction |
| AFM | IJCAI'17    | Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks |
| Wide & Deep | RecSys'16   | Wide & Deep Learning for Recommender Systems    
| DeepFM | IJCAI'17    | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction |
| NFM | SIGIR'17 | Neural Factorization Machines for Sparse Predictive Analytics  | 
| DCN | ADKDD'17    | Deep & Cross Network for Ad Click Predictions
| xDeepFM | KDD'18      | xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems | 
| FIGNN | CIKM '19 | Fi-GNN: Modeling Feature Interactions via Graph Neural Networks for CTR Prediction |    
| DCNv2 | WWW '21     | DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems |
| MaskNet | arxiv | Masknet: Introducing feature-wise multiplication to CTR ranking models by instance-guided mask |
| FinalMLP | AAAI'23 | FinalMLP: an enhanced two-stream MLP model for CTR prediction |
| EulerNet | SIGIR'23 | Eulernet: Adaptive feature interaction learning via euler's formula for ctr prediction |
| WuKong | ICML'24 | Wukong: Towards a scaling law for large-scale recommendation |

### Recall Task
#### General Models
| Model | Publish | paper name |
| :--- | :--- | :--- |
| BPR | UAI'09 | BPR: Bayesian personalized ranking from implicit feedback |

#### Multimodel Models

| Model | Publish | paper name |
| :--- | :--- | :--- |
| VBPR | AAAI'16 | VBPR: visual Bayesian Personalized Ranking from implicit feedback |
| FREEDOM | MM'23 | A tale of two graphs: Freezing and denoising graph structures for multimodal recommendation |
| LGMRec | AAAI'24 | Lgmrec: Local and global graph learning for multimodal recommendation |
<!-- | Model     | Publish     | Paper                                                        |
| :-------- | :---------- | :----------------------------------------------------------- |
| AFM       | IJCAI'17    | Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks |
| DCN       | ADKDD'17    | Deep & Cross Network for Ad Click Predictions                |
| DCN V2    | WWW '21     | DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems |
| DeepFM    | IJCAI'17    | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction |
| FM        | ICDM'10     | Factorization Machines                                       |
| FFM       | RecSys'16   | Field-aware Factorization Machines for CTR Prediction        |
| WideDeep  | RecSys'16   | Wide & Deep Learning for Recommender Systems                 |
| xDeepFM   | KDD'18      | xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems | -->

## Audio Featrues

We use pre-trained models to extract high-level music information. The following are the models we used:

| Model     | Publish     | Paper | 
| :-------- | :---------- |   :----------------------------------------------------------- | 
| [CLAP](https://github.com/microsoft/clap)      | ICASSP'22    | Natural Language Supervision For General-Purpose Audio Representations |
| [MuQ](https://github.com/tencent-ailab/MuQ)       | arxiv    | MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization  |

The MuQ model has two versions: **MuQ** and **MuQ-mulan**.

* **MuQ** provides the model's raw output, which includes both the number of layers and the time dimension. 
* **MuQ-mulan** is a fine-tuned version of MuQ, trained specifically on a music-text dataset, and its output has a shape of (512,).

## Results

### CTR overall
| Model | m4a | | | lfm-2b-taste | | | lfm-1b-taste | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | w/o multimodal | MuQ | MuQ-token | w/o multimodal | MuQ | MuQ-token | w/o multimodal | MuQ | MuQ-token |
| | **IDonly** | | | | | | | | |
| LR | 78.31 | 78.25 | 78.26 | 81.27 | 81.21 | 81.21 | 80.78 | 80.73 | 80.74 |
| FM | 78.83 | 80.92 | 80.22 | 85.17 | 85.25 | 84.82 | 84.48 | 84.67 | 84.17 |
| FFM | 79.25 | 79.33 | 80.05 | 84.06 | 84.04 | 83.97 | 83.34 | 83.45 | 83.83 |
| AFM | 79.85 | 81.48 | 82.80 | 85.12 | 85.47 | 85.83 | 84.90 | 84.78 | 85.17 |
| FiGNN | 79.69 | 81.50 | 81.99 | 84.31 | 85.08 | 85.74 | 83.87 | 84.24 | 84.84 |
| WideDeep | 79.94 | 81.53 | 81.56 | 84.02 | 84.84 | 86.00 | 84.58 | 84.78 | 85.45 |
| DeepFM | 79.79 | 81.78 | 81.26 | 83.81 | 84.06 | 83.86 | 83.43 | 83.16 | 83.37 |
| NFM | 79.77 | 82.54 | 82.64 | 83.32 | 85.85 | 86.02 | 82.24 | 83.87 | 85.23 |
| xDeepFM | 79.30 | 82.21 | 82.29 | 82.03 | 85.52 | 85.84 | 83.43 | 83.79 | 85.18 |
| DCN | 80.39 | 82.14 | 82.41 | 85.22 | 85.90 | 86.62 | 85.37 | 85.23 | 86.18 |
| DCNv2 | 80.03 | 82.36 | 82.97 | 85.60 | 86.58 | **86.89** | 85.36 | 85.55 | **86.21** |
| MaskNet | 79.88 | 82.92 | **83.22** | 84.78 | 86.57 | 86.68 | 83.85 | 85.23 | 85.93 |
| FinalMLP | 79.88 | 81.69 | 81.45 | 85.79 | 86.29 | 86.53 | 85.05 | 85.05 | 85.95 |
| EulerNet | 80.45 | 81.35 | 82.57 | 85.01 | 85.64 | 86.53 | 85.40 | 85.57 | 85.96 |
| WuKong | 79.10 | 82.34 | 82.54 | 83.64 | 84.57 | 84.78 | 82.72 | 84.11 | 84.95 |
| | **ID+Categories** | | | | | | | | |
| LR | 78.30 | 78.24 | 78.26 | 81.27 | 81.24 | 81.20 | 80.78 | 80.73 | 80.74 |
| FM | 79.65 | 80.85 | 80.23 | 85.26 | 85.33 | 84.82 | 84.61 | 84.55 | 84.22 |
| FFM | 79.49 | 79.55 | 80.49 | 83.50 | 84.22 | 84.41 | 83.98 | 83.95 | 83.98 |
| AFM | 80.10 | 81.96 | 82.86 | 85.41 | 85.89 | 86.26 | 84.62 | 85.23 | 85.49 |
| FiGNN | 80.45 | 81.96 | 81.99 | 84.21 | 85.94 | 85.91 | 84.41 | 84.89 | 85.16 |
| WideDeep | 80.56 | 81.86 | 81.86 | 84.50 | 85.87 | 86.17 | 84.88 | 84.99 | 85.57 |
| DeepFM | 80.46 | 81.86 | 81.46 | 84.11 | 84.29 | 83.85 | 83.37 | 83.68 | 83.56 |
| NFM | 80.72 | 82.60 | 82.79 | 85.08 | 86.19 | 86.38 | 83.82 | 85.48 | 85.55 |
| xDeepFM | 79.90 | 82.73 | 82.83 | 85.21 | 85.60 | 86.12 | 84.56 | 85.25 | 85.24 |
| DCN | 80.93 | 82.34 | 82.77 | 85.87 | 86.64 | 86.87 | 85.87 | 85.93 | 86.41 |
| DCNv2 | 81.04 | 82.87 | 83.10 | 86.54 | 86.91 | **87.10** | 85.81 | 86.00 | **86.47** |
| MaskNet | 80.82 | 83.28 | **83.66** | 85.86 | 87.00 | 87.07 | 85.05 | 86.07 | 86.29 |
| FinalMLP | 80.41 | 82.07 | 81.83 | 85.55 | 86.53 | 86.82 | 85.58 | 85.67 | 86.15 |
| EulerNet | 81.39 | 82.08 | 82.61 | 85.64 | 86.55 | 86.93 | 85.75 | 85.73 | 86.25 |
| WuKong | 80.16 | 82.41 | 82.70 | 84.66 | 85.02 | 85.31 | 82.97 | 83.01 | 83.22 |
| | **ID+Categories+Numeric** | | | | | | | | |
| LR | 78.34 | 78.30 | 78.34 | 81.27 | 81.24 | 81.20 | 80.70 | 80.70 | 80.66 |
| FM | 79.84 | 80.58 | 80.38 | 85.26 | 85.33 | 84.82 | 84.34 | 84.29 | 84.11 |
| FFM | 80.34 | 80.35 | 80.92 | 84.22 | 84.25 | 84.27 | 85.53 | 85.48 | 85.15 |
| AFM | 80.37 | 82.28 | 82.96 | 85.41 | 85.89 | 86.26 | 86.47 | 86.45 | 86.75 |
| FiGNN | 80.92 | 82.42 | 81.86 | 84.96 | 85.94 | 85.91 | 85.49 | 85.72 | 85.91 |
| WideDeep | 81.26 | 81.83 | 82.09 | 85.38 | 85.63 | 85.92 | 85.58 | 85.62 | 85.97 |
| DeepFM | 80.94 | 81.86 | 81.39 | 84.11 | 84.24 | 83.93 | 83.72 | 83.78 | 83.40 |
| NFM | 81.25 | 82.31 | 82.98 | 85.10 | 86.19 | 86.42 | 85.78 | 86.00 | 86.50 |
| xDeepFM | 80.51 | 81.83 | 82.41 | 85.21 | 86.07 | 86.13 | 85.70 | 85.87 | 86.29 |
| DCN | 81.41 | 82.64 | 82.91 | 86.43 | 86.73 | 86.88 | 86.75 | 86.76 | 86.92 |
| DCNv2 | 81.52 | 82.62 | 83.32 | 86.47 | 87.01 | 87.03 | 86.81 | 86.81 | 87.15 |
| MaskNet | 81.63 | 83.17 | **83.61** | 85.94 | 87.01 | **87.11** | 86.98 | 87.00 | **87.19** |
| FinalMLP | 81.17 | 82.15 | 82.06 | 86.25 | 86.55 | 86.70 | 86.40 | 86.36 | 86.66 |
| EulerNet | 81.43 | 82.54 | 82.66 | 86.44 | 86.56 | 86.68 | 86.83 | 86.88 | 86.93 |
| WuKong | 81.22 | 82.13 | 82.45 | 85.10 | 85.36 | 85.44 | 85.33 | 85.72 | 85.94 |
### CTR Cold-start 

**Music4all Dataset**

| Model | All | | All+MuQ-token | |
| :--- | :--- | :--- | :--- | :--- |
| | AUC(%) $\uparrow$ | logloss(%) $\downarrow$ | AUC(%) $\uparrow$ | logloss(%) $\downarrow$ |
| EulerNet | 60.67 | 0.4931 | 61.27 | 0.4824 |
| FinalMLP | 59.29 | 0.4965 | 62.09 | 0.4690 |
| DCNv2 | 62.89 | 0.5235 | 65.11 | 0.4840 |
| MaskNet | 59.08 | 0.5154 | 64.54 | 0.4767 |
| AFM | 63.84 | 0.4687 | 64.92 | 0.4548 |

**lfm-2b-taste Dataset**

| Model | All | | All+MuQ-token | |
| :--- | :--- | :--- | :--- | :--- |
| | AUC(%) $\uparrow$ | logloss(%) $\downarrow$ | AUC(%) $\uparrow$ | logloss(%) $\downarrow$ |
| EulerNet | 53.82 | 1.0539 | 60.60 | 0.9780 |
| FinalMLP | 50.41 | 0.9099 | 56.38 | 0.8140 |
| DCNv2 | 56.00 | 1.4961 | 59.90 | 1.4064 |
| MaskNet | 51.11 | 0.9479 | 59.17 | 0.8813 |
| AFM | 52.04 | 0.9047 | 56.80 | 0.8233 |
## Acknowledgement
<!-- We sincerely appreciate the help provided by [Recbole](https://github.com/RUCAIBox/RecBole). -->

We gratefully acknowledge the inspiration from frameworks [Recbole](https://github.com/RUCAIBox/RecBole), [MMRec](https://github.com/enoche/MMRec), and [FuxiCTR](https://github.com/reczoo/FuxiCTR).

## Future
We are working on adding more existing models, especially multimodal models.
