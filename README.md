# SegCassini
Segmentation on Cassini maps

## Data
### The data for retraining the model: 

https://drive.google.com/drive/folders/1py4qfVEpmBuTD0LK3aB6WQLupcwlzskr?usp=sharing

### The pretrained models are given in 

https://drive.google.com/drive/folders/1jvUlRE6itKqgGAQstv6SrXccVTNCR7Il?usp=sharing

## Segmentation on Cassini maps
To evaluate:

the data is given in 

https://drive.google.com/drive/folders/1IX-uXjQgdKK9mllS2Oi-Q21xTkCFcTy_?usp=sharing

run
    ```python segment_cassini_test.py```

## Style transfer
To train the model for style transfer:
    ```python cyclegan_qgis.py```

To use the denoiser:
    ```python cyclegan_qgis_dncnn.py```

## Segmentation model on IGN maps
To train a segmentation model on IGN maps:
    ```python segment_train.py```

To test:
    ```python segment_test.py```

## Baseline model
To train the baseline segmentation model on Cassini maps by using the pseudo labels generated from IGN maps as groundtruth:
    ```python segment_cassini_baseline_train.py```

To test
    ```python segment_cassini_baseline_test.py```



