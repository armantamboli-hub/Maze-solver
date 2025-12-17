GAN Architecture

The system is based on a Generative Adversarial Network (GAN) consisting of two neural networks: a Generator and a Discriminator.

The Generator learns to produce realistic images from random noise vectors, while the Discriminator acts as a binary classifier that distinguishes real images from generated ones.

Model Initialization

Both the Generator and Discriminator are explicitly initialized prior to training.

A custom initialize_weights function is applied to all trainable layers to ensure stable learning and proper parameter distribution, reducing the risk of gradient-related issues.

Data Pipeline

The dataset is loaded using torchvision.datasets.ImageFolder, with images stored in the moon_dataset directory.

Image transformations are applied as part of preprocessing to standardize input data.

A PyTorch DataLoader is used to enable efficient batch processing, shuffling, and parallel data loading.

Training Strategy

Training follows an adversarial learning paradigm where the Generator and Discriminator are optimized simultaneously with opposing objectives.

Both models use the Adam optimizer with carefully tuned beta parameters to improve convergence stability.

Binary Cross-Entropy Loss (nn.BCELoss) is employed to quantify the Discriminator’s classification error and guide the Generator’s learning process.

Checkpointing and Persistence

Model checkpoints are saved using PyTorch’s torch.save functionality.

Each checkpoint includes the state dictionaries of both models, optimizer states, and the current training epoch.

Checkpoints can be restored using torch.load, allowing training to resume seamlessly and enabling evaluation at different training stages.
