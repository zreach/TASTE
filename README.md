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

After the audio features are extracted, they pass through trainable MLP layers and are then merged with the previously extracted features to be input into various models, which ultimately output the predicted click-through rate.

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

## Results
### Overall Results

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

### Cold-start Results

| Model     | All AUC(%) ↑ | All logloss(%) ↓ | All+CLAP AUC(%) ↑ | All+CLAP logloss(%) ↓ | All+MuQ AUC(%) ↑ | All+MuQ logloss(%) ↓ |
| :-------- | ------------ | ---------------- | ----------------- | --------------------- | ---------------- | -------------------- |
| FM        | 69.16        | 52.56            | 71.26             | 53.18                 | 72.73            | 52.25                |
| WideDeep  | 72.65        | 54.44            | 72.76             | 54.42                 | 73.36            | 54.83                |
| DeepFM    | 73.63        | **51.96**        | **73.57**         | **52.21**             | **75.02**        | **51.69**            |
| NFM       | 69.11        | 55.83            | 69.03             | 59.48                 | 68.04            | 57.85                |
| AFM       | 66.69        | 52.48            | 65.04             | 52.97                 | 66.29            | 52.58                |
| xDeepFM   | 72.18        | 56.90            | 72.01             | 57.18                 | 74.79            | 51.98                |
| DCN       | 73.86        | 53.05            | 73.16             | 53.32                 | 74.21            | 52.22                |
| DCNv2     | **74.07**    | 53.27            | 73.54             | 53.58                 | 74.16            | 53.33                |

## Acknowledgement
We sincerely appreciate the help provided by [Recbole](https://github.com/RUCAIBox/RecBole).

## Future
We are working on adding more existing models, especially multimodal models.
