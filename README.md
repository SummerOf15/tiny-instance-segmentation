# tiny-instance-segmentation

## Method
### Detection

The resnet50+fpn network is applied to detect tufts from images.

### Classification

A image registration method is used to connect the reference image and target image, then it propagate class information between these images.

### Segmentation

A curve segmentation method is used to segment the tufts.

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