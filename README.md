# 🐱 Cats-GAN: DCGAN for Cat Faces (32x32, PyTorch)

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** in **PyTorch** to generate realistic 32x32 images of cat faces. It is designed to run seamlessly on Kaggle, using the [cats-faces-64x64-for-generative-models]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models)) dataset (downsampled to 32x32).

---

## 🧠 Model Overview

- **Generator:** Converts a 128-dimensional random vector (noise) into a 32x32 RGB cat image using transposed convolutions.
- **Discriminator:** CNN that classifies images as real (from dataset) or fake (from generator).
- **Loss Function:** Binary Cross Entropy (BCE) for both networks.

---

## 🚀 Quick Start (Kaggle Notebook)

**1. Enable GPU:**  
Click ⚙️ **Settings** (top right) and set **Accelerator** to **GPU**.

**2. Attach Dataset:**  
Add the dataset:  
`/kaggle/input/cats-faces-64x64-for-generative-models`

**3. Run the Notebook:**  
- All code and dependencies (`torch`, `torchvision`, `matplotlib`, `tqdm`) are available by default.
- All images are resized and center-cropped to **32x32**.
- Only the **first 10,000 images** are used for faster training.

---

## 🏗️ Main Features in the Code

- **Image Preprocessing:**  
  - Resize & crop images to 32x32  
  - Normalize to `[-1, 1]` for `tanh`-activated generator
- **Efficient DataLoader:**  
  - Uses only the first 10,000 images via `Subset`
- **Device Support:**  
  - Seamless GPU/CPU switching with custom `DeviceDataLoader`
- **Generator & Discriminator:**  
  - Updated architectures (32x32) to match code
  - Generator: latent vector → (4x4) → (8x8) → (16x16) → (32x32)
  - Discriminator: (32x32) → (16x16) → (8x8) → (4x4) → real/fake score
- **Training Loop:**  
  - Losses, scores, and checkpoints are tracked
  - Models auto-load from checkpoints if present
- **Visualization:**  
  - Functions to display real images, generated images, and loss curves

---

## 📸 Output Example

### 1. **Sample Cat Images (real)**
*Generated with `show_batch(cat_loader)`*

![Sample Cat Images (real)](cat_real_sample.png)

### 2. **Sample Generated Images (fake)**
*Generated with `reproduce()` after training*

![Sample Generated Cat Images (fake)](reproduced_images.png)

---

## 📝 Notes

- Only **10,000** images are loaded to keep training fast.
- **All images** are normalized to `[-1, 1]` as required by `tanh`.
- Generator/discriminator architectures are customized for **32x32** images.
- Training progress and loss curves are displayed after every 5 epochs.

---

## ✍️ Authors

- Nagham Omar  
- Zina Assi  
- Kater Alnada Watted  


---

**Enjoy generating cat faces!** 🐾
