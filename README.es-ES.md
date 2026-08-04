# TASTE
<div align="center">
    <img src="pics/logo.png" alt="logo" width="400">
</div>



TASTE es un marco para la recomendación de música con contenido aumentado. Fue desarrollado para reproducir y desarrollar algoritmos de recomendación con contenido de manera más eficiente.

El subconjunto propuesto de lfm-1b y las características embebidas de audio wav correspondientes pueden descargarse aquí: [Google Drive](https://drive.google.com/drive/folders/1H-wrqchl-QMWrO-13mueeO5t-7nL00JU?usp=sharing) [NJU Box](https://box.nju.edu.cn/d/dcbddef7b3624fd5beab/)

## Resumen

<div align="center">
    <img src="pics/framework.png" alt="framework" width="800">
    <p>Figura : Visión general del marco TASTE para la recomendación de música con contenido aumentado.</p>
</div>

Este marco incluye extracción de características, fusión de características, entrenamiento y prueba del modelo. 

Las características tradicionales se procesan de la manera habitual: las características discretas se procesan usando codificación one-hot, luego se mapean a vectores continuos de menor dimensionalidad; las características continuas se discretizan según el método definido por defecto, y luego se tratan como características discretas (se pueden usar otros métodos para manejar características continuas).

Después de que las características de audio y texto se extraen, pasan a través de capas MLP entrenables y luego se fusionan con las características previamente extraídas para ser introducidas en varios modelos, que finalmente generan la tasa de clics prevista.

Además, aplicamos el método **MuQ-token** a la salida multi-capa de nuestro modelo "MuQ". Al usar tokens discretos para representar las características de audio, logramos resultados mejores.

Nuestro método es altamente compatible con varios modelos porque solo agrega más características sin requerir cambios en la estructura del modelo en sí.

## Uso

Clona este repositorio a la máquina local, luego instala las dependencias.
```
pip install -r requirements.txt
```

Si deseas extraer manualmente las características de audio basándote en tus propios datos y modelo, puedes usar el archivo `./notebook/extract_feature.ipynb` en el directorio `./notebooks/`. Para obtener instrucciones detalladas, consulta los archivos de script.

### Inicio Rápido

Después de descargar los datos, colócalos en `datas\lfm1b-filtered` y luego ejecuta el siguiente comando:

```
python main.py
```

Este comando ejecuta el marco con los ajustes predeterminados de la manera más sencilla. Si necesitas ajustar los parámetros después, pueden establecerse ya sea a través de la línea de comandos o utilizando archivos YAML, por ejemplo:

```
python main.py --model_name LR --dataset_name lfm1n-filtered ----config_files config/config1.yaml
```

## Modelos

Actualmente, hemos implementado los siguientes modelos en TASTE:

### Tarea CTR
#### Modelos conscientes del contexto

| Modelo | Publicación | Nombre del artículo |
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

### Tarea Recall
#### Modelos Generales
| Modelo | Publicación | Nombre del artículo |
| :--- | :--- | :--- |
| BPR | UAI'09 | BPR: Bayesian personalized ranking from implicit feedback |

#### Modelos Multimodales

| Modelo | Publicación | Nombre del artículo |
| :--- | :--- | :--- |
| VBPR | AAAI'16 | VBPR: visual Bayesian Personalized Ranking from implicit feedback |
| FREEDOM | MM'23 | A tale of two graphs: Freezing and denoising graph structures for multimodal recommendation |
| LGMRec | AAAI'24 | Lgmrec: Local and global graph learning for multimodal recommendation |

## Características de Audio

Usamos modelos preentrenados para extraer información musical de alto nivel. A continuación, se muestran los modelos que usamos:

| Modelo     | Publicación     | Artículo | 
| :-------- | :---------- |   :----------------------------------------------------------- | 
| [CLAP](https://github.com/microsoft/clap)      | ICASSP'22    | Natural Language Supervision For General-Purpose Audio Representations |
| [MuQ](https://github.com/tencent-ailab/MuQ)       | arxiv    | MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization  |

El modelo MuQ tiene dos versiones: **MuQ** y **MuQ-mulan**.

* **MuQ** proporciona la salida cruda del modelo, que incluye tanto el número de capas como la dimensión temporal.  
* **MuQ-mulan** es una versión afinada de MuQ, entrenada específicamente en un conjunto de datos música-texto, y su salida tiene una forma de (512,).

## Resultados

### CTR general
| Modelo | m4a | | | lfm-2b-taste | | | lfm-1b-taste | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | sin multimodal | MuQ | MuQ-token | sin multimodal | MuQ | MuQ-token | sin multimodal | MuQ | MuQ-token |
| | **SoloID** | | | | | | | | |
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
| | **ID+Categorías** | | | | | | | | |
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
| EulerNet | 81.39 | 82.08 | 82.61 | 86.44 | 86.56 | 86.68 | 86.83 | 86.88 | 86.93 |
| WuKong | 80.16 | 82.41 | 82.70 | 84.66 | 85.02 | 85.31 | 82.97 | 83.01 | 83.22 |
| | **ID+Categorías+Númericos** | | | | | | | | |
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

**Conjunto de datos Music4all**

| Modelo | Todos | | Todos+MuQ-token | |
| :--- | :--- | :--- | :--- | :--- |
| | AUC(%) $\uparrow$ | logloss(%) $\downarrow$ | AUC(%) $\uparrow$ | logloss(%) $\downarrow$ |
| EulerNet | 60.67 | 0.4931 | 61.27 | 0.4824 |
| FinalMLP | 59.29 | 0.4965 | 62.09 | 0.4690 |
| DCNv2 | 62.89 | 0.5235 | 65.11 | 0.4840 |
| MaskNet | 59.08 | 0.5154 | 64.54 | 0.4767 |
| AFM | 63.84 | 0.4687 | 64.92 | 0.4548 |

**Conjunto de datos lfm-2b-taste**

| Modelo | Todos | | Todos+MuQ-token | |
| :--- | :--- | :--- | :--- | :--- |
| | AUC(%) $\uparrow$ | logloss(%) $\downarrow$ | AUC(%) $\uparrow$ | logloss(%) $\downarrow$ |
| EulerNet | 53.82 | 1.0539 | 60.60 | 0.9780 |
| FinalMLP | 50.41 | 0.9099 | 56.38 | 0.8140 |
| DCNv2 | 56.00 | 1.4961 | 59.90 | 1.4064 |
| MaskNet | 51.11 | 0.9479 | 59.17 | 0.8813 |
| AFM | 52.04 | 0.9047 | 56.80 | 0.8233 |
## Reconocimiento
<!-- Agradecemos sinceramente la ayuda proporcionada por [Recbole](https://github.com/RUCAIBox/RecBole). -->

Agradecemos el inspiración de los marcos [Recbole](https://github.com/RUCAIBox/RecBole), [MMRec](https://github.com/enoche/MMRec), y [FuxiCTR](https://github.com/reczoo/FuxiCTR).

## Futuro
Estamos trabajando en agregar más modelos existentes, especialmente modelos multimodales.
