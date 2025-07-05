# 🐱 Cats-GAN: DCGAN for Cat Faces

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using **PyTorch** to generate realistic images of cat faces. The model is trained on a dataset of 64x64 cat images.

## 🧠 Model Overview

* **Generator**: Uses transposed convolution layers to generate 64x64 RGB cat images from a 128-dimensional latent vector.
* **Discriminator**: A CNN that distinguishes between real cat images and those produced by the generator.
* **Loss Function**: Binary Cross Entropy (BCE) is used for both generator and discriminator.

## 🚀 Running the Notebook on Kaggle

### 1. **Enable GPU**

Go to **Settings** (⚙️ on the top-right of your Kaggle Notebook) and:

* Turn **Accelerator** to **GPU**
* Ensure **Internet** is turned off (dataset is assumed to be attached)

### 2. **Attach Dataset**

Attach the Kaggle dataset:
📁 `cats-faces-64x64-for-generative-models`
This dataset will be loaded from:

```python
/kaggle/input/cats-faces-64x64-for-generative-models
```

### 3. **Run the Notebook**

Run all cells in order. The training and generation process includes:

* Loading and preprocessing the dataset
* Visualizing real cat images
* Defining and training DCGAN (generator + discriminator)
* Saving and displaying generated images

## 📦 Dependencies

The Kaggle environment already includes most dependencies:

* `torch`
* `torchvision`
* `matplotlib`
* `tqdm`

Just make sure to run it on **GPU** for faster training.

## 📸 Sample Output

The model will output generated images such as:

```
Sample Cat Images (real)
Sample Generated Images (fake)
```

These will be displayed as matplotlib image grids.

## 📝 Notes

* Only the first 10,000 images from the dataset are used to reduce training time.
* Image normalization is performed to `[-1, 1]` as required by `tanh` activation in the generator.

## ✍️ Authors
  
Nagham Omar
  
Zina Assi
  
Kater Alnada Watted
  
Project developed as part of a GAN learning exercise.

---
