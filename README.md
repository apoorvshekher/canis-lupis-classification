# Canis Lupis Familiaris Breed Prediction
  
As the most popular domesticated species, Dogs or Canis lupus familiaris, are unique in terms of traits, features, and diverse physical and mental strength. Due to this ambiguous and distinctive nature, there are several notable concerns, including increasing population, disease transmission, adoption, and inoculation. To address the above-stated issues, there is a necessity to determine the species. The research aims to present a state-of-the-art classification approach using an image processing technique based on deep convolutional neural networks and transfer learning.

## Dataset Used 

The [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization.

![Sample Image](https://raw.githubusercontent.com/apoorvshekher/canis-lupis-classification/main/appendix/sample.png)

## Setup

1. Create a conda environment

```
conda create -n canis python=3.10
```

2. Clone this repository.

```
git clone git@github.com:apoorvshekher/canis-lupis-classification.git
```

3. Go to the directory and download dataset.

```
cd canis-lupis-familiaris
bash data/extract.sh
```
4. Perform training and testing

```
python .
```