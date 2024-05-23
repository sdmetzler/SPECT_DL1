import torch
import matplotlib.pyplot as plt

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# Load the three tensors from disk
file_path = '/Users/metzler/Work/PyCharmProjects/PlayNeuralNetworks/NewResults/'


file_prefix = 'validate'
fname_x = file_path + file_prefix + '_x.tar'  # Adjust the filenames accordingly
fname_y = file_path + file_prefix + '_y.tar'
fname_p = file_path + file_prefix + '_p.tar'

X_validate = torch.load(fname_x, map_location=device).view(-1, 1, 256, 256)
print(f"X_validate: {X_validate.shape}")
y_validate = torch.load(fname_y, map_location=device)
print(f"y_validate: {y_validate.shape}")
y_pred = torch.load(fname_p, map_location=device)
print(f"y_pred: {y_pred.shape}")

# Check the shape of the tensors
assert X_validate.shape == y_validate.shape == y_pred.shape, "Tensor shapes should be the same"

N, _, A, _ = X_validate.shape

# Loop over N and display images
print(f"There are {N} image sets.")
for n in range(N):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    axs[0].imshow(X_validate[n, 0].numpy(), cmap='gray')
    axs[0].set_title('X_validate')

    axs[1].imshow(y_validate[n, 0].numpy(), cmap='gray')
    axs[1].set_title('y_validate')

    axs[2].imshow(y_pred[n, 0].numpy(), cmap='gray')
    axs[2].set_title('y_pred')

    plt.show()

    # Wait for carriage return to proceed to the next image
    input("Press Enter to continue to the next image...")
