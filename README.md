# Cat Image Generation with GAN

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation for generating realistic cat images using PyTorch.

## Project Overview

This project implements a DCGAN to generate synthetic cat images from random noise. The model learns to generate high-quality 64x64 RGB cat images by training a generator network against a discriminator network in an adversarial setup.

## Features

- **DCGAN Architecture**: Implements the proven Deep Convolutional GAN architecture
- **64x64 RGB Output**: Generates high-quality color images
- **Kaggle Integration**: Designed to run on Kaggle with GPU acceleration
- **AFHQ Dataset**: Uses the Animal Faces-HQ (AFHQ) lite dataset for training
- **Proper Normalization**: Images normalized to [-1, 1] range for optimal GAN training

## Dataset

The project uses the AFHQ (Animal Faces-HQ) lite dataset, which contains high-quality animal face images. The dataset should be structured as:

```
data/
└── afhq_lite/
    └── cat/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Model Architecture

### Generator
- **Input**: 100-dimensional noise vector
- **Architecture**: 5 transposed convolutional layers with batch normalization
- **Output**: 64x64x3 RGB images
- **Activation**: ReLU (hidden layers), Tanh (output layer)

### Discriminator
- **Input**: 64x64x3 RGB images
- **Architecture**: 5 convolutional layers with batch normalization
- **Output**: Single probability value (real/fake)
- **Activation**: LeakyReLU (hidden layers), Sigmoid (output layer)

## Requirements

```python
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
```

## Installation & Setup

1. **Clone or download** the project files
2. **Upload to Kaggle** or set up in your preferred environment
3. **Add the AFHQ dataset** to your Kaggle notebook or local environment
4. **Enable GPU acceleration** (recommended for faster training)

## Usage

### Basic Setup

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
])

# Load dataset
dataset = datasets.ImageFolder(root='data/afhq_lite', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### Model Initialization

```python
# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create models
netG, netD = create_models(device)

# Setup training
criterion, optimizerD, optimizerG = setup_training(netG, netD)
```

### Training Loop

The training follows the standard GAN training procedure:

1. **Train Discriminator**: 
   - Update discriminator with real images (label=1)
   - Update discriminator with fake images (label=0)

2. **Train Generator**:
   - Generate fake images and try to fool discriminator (label=1)

## Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `nz` | 100 | Size of noise vector |
| `ngf` | 64 | Generator feature map size |
| `ndf` | 64 | Discriminator feature map size |
| `lr` | 0.0002 | Learning rate |
| `beta1` | 0.5 | Adam optimizer beta1 parameter |
| `batch_size` | 64 | Training batch size |
| `image_size` | 64 | Output image dimensions |

## File Structure

```
cat-gan-project/
├── cats-gan.ipynb          # Main Jupyter notebook
├── README.md              # This file
└── models/
    ├── generator.py       # Generator model definition
    ├── discriminator.py   # Discriminator model definition
    └── utils.py          # Helper functions
```

## Training Tips

1. **Monitor Loss**: Both generator and discriminator losses should decrease over time
2. **Learning Rate**: Use 0.0002 for stable training
3. **Batch Size**: 64 works well for most setups
4. **Epochs**: Train for 100-500 epochs depending on dataset size
5. **Save Checkpoints**: Save model states regularly during training

## Expected Results

After successful training, the generator should produce realistic cat images that:
- Have proper cat facial features (ears, eyes, whiskers)
- Show variety in colors and patterns
- Maintain good image quality at 64x64 resolution

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure dataset path is correct
2. **CUDA Out of Memory**: Reduce batch size or use CPU
3. **Mode Collapse**: Try different learning rates or architectures
4. **Poor Quality**: Increase training epochs or adjust hyperparameters

### Dataset Path Issues

If you encounter path errors, verify:
- Dataset is uploaded to correct location
- Folder structure matches expected format
- File permissions are correct (in local environments)

## Performance

- **Training Time**: ~2-4 hours on GPU (depends on dataset size)
- **Memory Usage**: ~4-8GB GPU memory
- **Model Size**: ~50MB for both generator and discriminator

## Future Improvements

- [ ] Implement progressive growing for higher resolution outputs
- [ ] Add StyleGAN features for better control
- [ ] Implement FID score evaluation
- [ ] Add data augmentation techniques
- [ ] Support for different animal classes

## License

This project is open source and available under the MIT License.

## Acknowledgments

- DCGAN paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- AFHQ Dataset: [Stargan-v2 repository](https://github.com/clovaai/stargan-v2)
- PyTorch team for excellent deep learning framework

## Contact

For questions or issues, please open an issue in the project repository or contact the development team.

---

**Note**: This project is designed for educational purposes and research. Results may vary depending on dataset quality and training parameters.
