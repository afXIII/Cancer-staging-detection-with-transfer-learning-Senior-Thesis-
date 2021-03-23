# Cancer Metastasis using CNNs and transfer learning

## Links
Link to dataset: https://www.kaggle.com/c/histopathologic-cancer-detection/overview

Dataset is also available at:
```bash
cluster.cs.earlham.edu:/eccs/home/afarah18/488/markIII/histopathologic-cancer-detection
```

Link to Paper: https://portfolios.cs.earlham.edu/wp-content/uploads/2021/03/paper488.pdf

Link to Video:

Link to Poster: https://portfolios.cs.earlham.edu/wp-content/uploads/2021/03/poster.pdf

## Software architecture diagram
![Software architecture diagram](https://portfolios.cs.earlham.edu/wp-content/uploads/2021/03/Screen-Shot-2021-03-22-at-11.42.40-PM.png)

## Files

basicCNN.py -> This file includes code for training a model with randomly initilized weights and biases

VGGCNN.py -> This file includes code for training a VGG model with ImageNet transfer learning

VGGLessData.py -> This file includes code for training VGG with transfer learning with half of the dataset

validate.py -> This file includes code for evaluating the results of the three models

## How to use
After installing the dependencies each file can be run simply by using:
```bash
python3 <filename>
```

Naturally validation file has to be run after all three models have been trained and saved as .h5 files.

## List of dependencies
+ NumPy
+ Pandas
+ TensorFlow
+ Keras
+ MatplotLib