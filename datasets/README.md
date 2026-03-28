# Datasets

This project is prepared for a custom 10-class animal detector.

Expected training dataset root:

`datasets/animals10/`

Required YOLO detection layout:

- `datasets/animals10/images/train/`
- `datasets/animals10/images/val/`
- `datasets/animals10/images/test/`
- `datasets/animals10/labels/train/`
- `datasets/animals10/labels/val/`
- `datasets/animals10/labels/test/`

Each label file must use YOLO detection format:

`class_id x_center y_center width height`

All coordinates must be normalized to the image width and height.

Class order for this project:

0. dog
1. cat
2. cow
3. horse
4. deer
5. elephant
6. zebra
7. giraffe
8. tiger
9. lion
