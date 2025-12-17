GAN uses Generator and Discriminator, initialized with custom weight initialization.
• Images are loaded using ImageFolder from moon_dataset and batched via DataLoader.
• Training uses Adam optimizer with Binary Cross-Entropy Loss.
• Model, optimizer, and epoch states are saved and restored using torch.save and torch.load.
