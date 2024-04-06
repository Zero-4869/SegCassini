# SegCassini
Segmentation on Cassini maps

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

## Segmentation on Cassini maps
To evaluate:
    ```python aggregation_test.py```

## Baseline
To train the baseline segmentation model on Cassini maps by using the pseudo labels generated from IGN maps as groundtruth:
    ```python segment_cassini_baseline_train.py```

To test
    ```python segment_cassini_baseline_test.py```



