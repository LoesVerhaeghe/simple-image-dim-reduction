from src.autoencoder.model_structure import Autoencoder
from utils.helpers import extract_images
from src.autoencoder.training_autoencoder import train_autoencoder
from src.autoencoder.images_dataset import MicroscopicImages
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#load images
base_folder = "data/microscope_images"
## dataset maken!
#all_images = extract_images(base_folder, image_type='all', magnification=10)

# load model structure
model=Autoencoder()
print(summary(model, input_size=(3, 512, 512)))

# Setting training parameters
IMAGE_DIMENSION=(512, 512)
RANDOM_SEED= 42
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_CLASSES = 10
SKIP_EPOCH_STATS=False
SAVE_MODEL=False
LOSS_FN = nn.MSELoss()
OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_DIMENSION)     
    ])

torch.manual_seed(RANDOM_SEED)

dataset = MicroscopicImages(root=base_folder, transform=train_transform) 
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
trained_model=train_autoencoder(NUM_EPOCHS, model, OPTIMIZER, data_loader, LOSS_FN, 
                                skip_epoch_stats=False, save_model=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to display original and reconstructed images
def visualize_reconstruction(model, test_loader, num_images=5):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for testing/inference
        for batch in test_loader:
            # Get a batch of test images and move them to the device
            batch_data = batch.to(device)
            
            # Forward pass to get the reconstructed images
            reconstructed = model(batch_data)
            
            # Move tensors to CPU and convert to numpy for visualization
            batch_data = batch_data.cpu()
            reconstructed = reconstructed.cpu()

            print(batch_data.size())
            
            # Plot the original and reconstructed images
            fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
            for i in range(num_images):
                # Original images
                axes[0, i].imshow(batch_data[i].permute(1, 2, 0).numpy())  # Convert CHW to HWC
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstructed images
                axes[1, i].imshow(reconstructed[i].permute(1, 2, 0).numpy())  # Convert CHW to HWC
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')
                
            plt.show()
            break  # Only show for one batch

visualize_reconstruction(model, data_loader, num_images=5)



