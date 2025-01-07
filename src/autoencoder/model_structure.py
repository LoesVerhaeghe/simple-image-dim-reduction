import torch
import torch.nn as nn

# Defining Autoencoder model
# check literatuur voor gelijkaardige studies die ook niet focussen op orientatie van de features
# check boek over meer fundamentele info over autoencoders
class Autoencoder(nn.Module):
   def __init__(self):
       super(Autoencoder, self).__init__()

       self.encoder = nn.Sequential(
           # CONVOLUTION LAYER 
           nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), #stride staat default op 1, padding=1 => image verandert niet van grootte
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2), #bij opschuiven geen overlapping omdat je een stride van 2 hebt 
           # CONVOLUTION LAYER 
           nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),
           # CONVOLUTION LAYER 
           nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2)
       ) #de image wordt nu 8x verkleind
       # Latent space
       self.flatten = nn.Flatten()
       self.fc1 = nn.Linear(1 * 64 * 64, 200)  # 8 * 16 * 16 = 2048 (feature maps size after encoding)
       self.fc2 = nn.Linear(200, 1 * 64 * 64)
       
       self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 8, kernel_size=3, stride=2, padding=1, output_padding=1), #check documentatie
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1), #check documentatie
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
       
   def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 64, 64)  # Reshape to match the decoder's input size
        x = self.decoder(x)
        return x
