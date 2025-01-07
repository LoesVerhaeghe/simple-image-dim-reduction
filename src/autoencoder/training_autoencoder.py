import torch
import torch.nn as nn
import time

def train_autoencoder(num_epochs, model, optimizer, data_loader,
                         loss_fn=nn.MSELoss(),
                         skip_epoch_stats=False,
                         save_model=None):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log_dict = {'train_loss_per_epoch': []}
    start_time = time.time()

    # Training the autoencoder
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode (in for lus?)
        epoch_loss = 0  # Initialize epoch loss

        for batch in data_loader:
            # Move the batch to the device
            batch_data = batch[0].to(device)

            # Forward pass
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_data)

            # Backward pass and optimization
            optimizer.zero_grad()  # Reset gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters

            # Accumulate the loss for the current epoch
            epoch_loss += loss.item()

        if not skip_epoch_stats:
            model.eval()
            with torch.set_grad_enabled(False):  # save memory during inference
                    avg_loss = epoch_loss / len(data_loader) 
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
                    log_dict['train_loss_per_epoch'].append(avg_loss)
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    #if save_model is not None:
    #        torch.save(model.state_dict(), save_model)

    return model  # Return the trained model



    