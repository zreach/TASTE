# TASTE
<div align="center">
    <img src="pics/logo.png" alt="描述" width="400">
</div>



TASTE is a framework for content-augumented Music Recommendation. It is developed for reproducing and developing recommendation algorithms more efficiently. We sincerely appreciate the help provided by [Recbole](https://github.com/RUCAIBox/RecBole)

The subset of lfm-1b and corresponding embedded wav features can be downloaded here: [Google Drive](https://drive.google.com/drive/folders/1H-wrqchl-QMWrO-13mueeO5t-7nL00JU?usp=sharing)

## Usage

Clone this repository to the local machine, then install the dependencies.
```
pip install -r requirements.txt
```

## Quick-Start

After downloading the data, place it in `datas\lfm1b-filtered` and then run the following command:

```
python main.py
```

This command runs the framework with the default settings in the simplest way. If adjustments are needed afterward, parameters can be set either through the command line or by using YAML files，for example:

```
python main.py --model_name LR --dataset_name lfm1n-filtered ----config_files config/config1.yaml
```

The config files can be a list.
## Models

Currently, we have implemented the following models on TASTE:


| Model     | Publish     | Paper                                                        |
| :-------- | :---------- | :----------------------------------------------------------- |
| AFM       | IJCAI'17    | Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks |
| DCN       | ADKDD'17    | Deep & Cross Network for Ad Click Predictions                |
| DCN V2    | WWW '21     | DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems |
| DeepFM    | IJCAI'17    | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction |
| FM        | ICDM'10     | Factorization Machines                                       |
| FFM       | RecSys'16   | Field-aware Factorization Machines for CTR Prediction        |
| WideDeep  | RecSys'16   | Wide & Deep Learning for Recommender Systems                 |
| xDeepFM   | KDD'18      | xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems |

## Audio Featrues

We use pre-trained models to extract high-level music information. The following are the models we used:

| Model     | Publish     | Paper 
| :-------- | :---------- | :----------------------------------------------------------- |
| [CLAP](https://github.com/microsoft/clap)      | ICASSP'24    | Natural Language Supervision For General-Purpose Audio Representations |
| [MuQ](https://github.com/tencent-ailab/MuQ)       | arxiv    | MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization |

## Overall Results

| Model     | w/o audio | CLAP   | MuQ    | w/o audio | CLAP   | MuQ    | w/o audio | CLAP   | MuQ    |
| :-------- | --------- | ------ | ------ | --------- | ------ | ------ | --------- | ------ | ------ |
|           | ID-only   |        |        | ID+Categories |        |        | ID+Categories+Continuous |        |        |
| LR        | 63.07     | 63.11  | 63.31  | 63.08     | 63.07  | 63.10  | 63.11     | 63.12  | 63.15  |
| FM        | **80.64** | 79.59  | 80.45  | 80.96     | 81.01  | 81.16  | 81.40     | 80.45  | 81.60  |
| WideDeep  | 80.16     | 79.55  | 79.95  | 80.53     | 80.56  | 80.59  | 80.58     | 79.95  | 80.90  |
| DeepFM    | 78.73     | 80.15  | 80.58  | 81.13     | 81.15  | 81.46  | 81.01     | 80.58  | 81.40  |
| xDeepFM   | 77.51     | 78.55  | 80.58  | 81.29     | 81.78  | 81.61  | 81.11     | 81.22  | 81.74  |
| NFM       | 78.25     | 78.55  | 78.89  | 79.53     | 79.77  | 80.54  | 79.57     | 78.92  | 80.01  |
| AFM       | 79.97     | 78.92  | 79.01  | 80.69     | 78.87  | 79.38  | 80.42     | 79.01  | 80.19  |
| DCN       | 78.55     | 78.98  | 79.94  | 81.22     | 80.90  | 81.31  | 81.06     | 79.94  | 81.37  |
| DCNv2     | 80.33     | **80.82** | **81.23** | **82.27**  | **82.22**  | **82.45**  | **82.24**  | **82.27**  | **82.51**  |


## Future
We are working on adding more existing models, especially multimodal models.
