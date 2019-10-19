# Classification-of-satellite-image
Classification on satellite image are performed using Machine Learning and Deep Learning algorithms. The
algorithms are trained on the subset of NPU-RESISC45 dataset which consists of 12 scene
classes, with 700 images. The classification model performance is evaluated using accuracy,
precision, recall, F1-score and confusion matrix.

| Description                                                                                       | Accuracy  | Precision  | Recall  | F1 score |
|---------------------------------------------------------------------------------------------------|-----------|------------|---------|----------|
| 1. Bag-of-Visual-Words                                                                            | 50.5      | 52.41      | 50.5    | 48.52    |
| 2. VGG16(ImageNet) + FullyConnected                                                               | 87.83     | 88.20      | 87.83   | 87.93    |
| 3. VGG16(ImageNet) +SVM                                                                           | 82.06     | 82.06      | 82.06   | 82.06    |
| 4. FineTune VGG16(ImageNet) +FullyConnected                                                       | 88.61     | 89.45      | 88.61   | 88.61    |
| 5. FineTune VGG16(ImageNet) withdata augmentation +FullyConnected                                 | 90.94     | 91.36      | 90.94   | 90.95    |
| 6. FineTune VGG16(Places365) +FullyConnected                                                      | 85.00     | 85.93      | 85.00   | 85.00    |
| 7. FineTune VGG16(Places365) withdata augmentation +FullyConnected                                | 85.50     | 86.25      | 85.50   | 85.54    |
| 8. Ensemble model 1:BoVW +FineTuneVGG16(Places365) +FullyConnected                                | 89.28     | 89.43      | 89.28   | 89.30    |
| 9. Ensemble model 2:FineTune VGG16(ImageNet)withdata augmentation + FineTuneVGG16(Places365) +SVM | 81.61     | 82.00      | 81.61   | 81.62    |
