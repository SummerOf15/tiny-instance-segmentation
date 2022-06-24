# tiny-instance-segmentation
The report is available [here](https://elib.dlr.de/146059/1/zhang_ensta_thesis.pdf)

## Method
### Detection

The resnet50+fpn network is applied to detect tufts from images.

![image](https://user-images.githubusercontent.com/24192012/175655910-737ad9c8-3182-48b0-ac40-be35a226e866.png)

### Classification

A image registration method is used to connect the reference image and target image, then it propagate class information between these images.

![image](https://user-images.githubusercontent.com/24192012/175656040-9de5d9f2-47d1-43cc-9d86-85e00ac4b7b6.png)

### Segmentation

A curve segmentation method is used to segment the tufts.

![image](https://user-images.githubusercontent.com/24192012/175656160-32de364f-f41f-484b-a270-34e77ee9fefe.png)

## Code
Up to now, the detection and classification parts are integrated into one files.

- If you want to predict the results for one image, modify the `dataset_dir` and `output_dir` in `predict_one_image.py`, then run the file by

```bash
python predict_one_image.py
```

- if you want to predict the results for all the images in one directory, modify the modify the `dataset_dir` and `output_dir` in `predict_one_image.py`, then run the file by

```bash
python predict_images_in_dir.py
```

## Results
![image](https://user-images.githubusercontent.com/24192012/175656264-4c865e8e-6573-42f1-8c80-bd897e49c988.png)
