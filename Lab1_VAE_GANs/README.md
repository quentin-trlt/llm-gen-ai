# Lab1_VAE_GANs

Lab 1 of the LLM & Generative AI course: three image generation models are built and trained from scratch in PyTorch on FashionMNIST, an autoencoder, a variational autoencoder and a GAN. The models are then compared on reconstruction quality, latent space structure and sample diversity.

---

## Structure

```
Lab1_VAE_GANs/
├── data/
│   └── FashionMNIST/raw/          # downloaded automatically on first run
├── .gitignore
├── Lab1_VAE_GANs.ipynb
├── pyproject.toml
└── README.md
```

---

## How it works

### 1. Setup and data
- Selects the device automatically (`cuda`, then `mps`, then CPU).
- Loads FashionMNIST through `torchvision.datasets.FashionMNIST(root="./data", download=True)` with batches of 128.

### 2. Autoencoder
- Fully connected encoder (784 → 128 → 64) and decoder (64 → 128 → 784), trained for 20 epochs with MSE loss and Adam.
- Plots the loss curve, original vs reconstructed images, and a t-SNE projection of the latent codes for 1000 images.

### 3. Variational autoencoder
- Convolutional encoder that outputs a mean and a log-variance, sampling with the reparameterization trick, and a transposed-convolution decoder.
- Loss is binary cross-entropy plus KL divergence.
- Trained for 20 epochs, then again with a 50-dimensional latent space for 50 epochs, and the two latent spaces are compared.

### 4. GAN
- MLP generator mapping a 100-dimensional noise vector to a 28x28 image, and an MLP discriminator classifying real vs generated images, both trained with BCE loss and Adam (`lr=2e-4`).
- Trained for 25 epochs, then for 100 epochs, with generated samples compared between the two runs.

### 5. Report
- A final markdown cell compares the three models.

---

## Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate with `.\.venv\Scripts\Activate.ps1`.

> Tested with: Python 3.12

---

## Running the project

### Locally

```bash
jupyter lab Lab1_VAE_GANs.ipynb
```

Start Jupyter from this folder, or open the notebook in VS Code or PyCharm with `.venv` as the kernel. The dataset path is relative to the notebook directory.

The notebook's first cell runs `!pip install torch torchvision matplotlib scikit-learn`. It does nothing more once the project is installed and can be skipped.

---

## Data & external resources

FashionMNIST (60,000 training images, 28x28 grayscale, 10 clothing classes) is downloaded by torchvision into `data/FashionMNIST/raw/` on first run (about 80 MB including the `.gz` archives). The folder is gitignored and does not need to be shared. Training the GAN for 100 epochs takes a long time on CPU, so a GPU or Apple Silicon (`mps`) is recommended.
