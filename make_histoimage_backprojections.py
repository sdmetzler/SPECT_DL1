import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon


def make_histo_image(data, diameter, image_size):
    # get views
    num_views = data.shape[0]
    num_bins = data.shape[1]
    offset_bin = (image_size - num_bins) // 2
    start_row = (image_size - diameter) // 2

    # Initialize an empty image
    image = np.zeros((num_views, image_size, image_size), dtype=np.float32)

    # initialize mask
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.circle(mask, (image_size // 2, image_size // 2), diameter // 2, (1,), -1)

    # show mask
    #plt.imshow(mask, cmap='gray')
    #plt.title('Mask')
    #plt.show()

    # Loop over each view
    for view in range(num_views):
        # Extract data for the current view
        this_view = data[view]
        this_image = image[view]

        # Draw vertical lines with amplitudes from view_data
        for bin_idx, amplitude in enumerate(this_view):
            this_image[start_row:(start_row+diameter), offset_bin+bin_idx] = amplitude

        #fig, axs = plt.subplots(1, 2)
        #axs[0].imshow(this_image, cmap='gray')

        # Rotate the image by 3 degrees * view number
        rotation_matrix = cv2.getRotationMatrix2D((image_size // 2, image_size // 2), -3 * view, 1)
        image[view] = cv2.warpAffine(this_image, rotation_matrix, (image_size, image_size)) * mask
        #print(f"Sum of this_image: {np.sum(this_image)}.")
        #axs[1].imshow(this_image, cmap='gray')

        # show image
        #plt.imshow(this_image, cmap='gray')
        #plt.title('Image')
        #plt.show()


    #print(f"Sum of all images: {np.sum(image)}.")

    # show summed image
    #plt.imshow(np.sum(image, axis=0), cmap='gray')
    #plt.title('Summed Histoimages')
    #plt.show()

    # return result
    return image


def find_proj_files(directory):
    """
    Search a directory for files ending with '.proj'.

    Args:
        directory (str): The directory to search in.

    Returns:
        list: A list of paths to files ending with '.proj'.
    """
    # Construct the pattern for '*.proj' files
    pattern = os.path.join(directory, '*.proj')

    # Use glob to find files matching the pattern
    proj_files = glob.glob(pattern)

    return proj_files


def load_float32_binary_file(file_path):
    """
    Load a binary file containing float32 data and return the data as a NumPy array.

    Args:
        file_path (str): The full path to the binary file.

    Returns:
        numpy.ndarray: The data loaded from the binary file.
        str: The directory containing the file.
        str: The file name.
        str: The file name without the suffix.
    """
    # Load the binary file into a NumPy array
    data = np.fromfile(file_path, dtype=np.float32)

    # Get the directory, file name, and file name without the suffix
    directory = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    file_name_no_suffix = os.path.splitext(file_name)[0]

    return data, directory, file_name, file_name_no_suffix



if __name__ == "__main__":
    # Parameters
    num_views = 120
    num_bins = 250
    image_size = 256  # bins
    diameter = image_size - 2  # bins
    verbose = True
    input_directory = '/Users/metzler/Work/Python/SPECT_DL/TrainData/AttenProj/'
    output_directory = '/Users/metzler/Work/Python/SPECT_DL/'

    # get a list of all the files
    proj_files = find_proj_files(input_directory)
    print(proj_files)

    # loop over files
    for file_path in proj_files:
        data, directory, file_name, file_name_no_suffix = load_float32_binary_file(file_path)
        if verbose:
            print("File name: ", file_name)
            #print("File name without suffix: ", file_name_no_suffix)
            #print("Data: ", data.shape)
            #print("Directory: ", directory)
        assert data.shape[0] == num_bins * num_views
        data = np.reshape(data, (num_views, num_bins))
        the_histo_image = make_histo_image(data, diameter, image_size)

        # show phantom
        #data_phantom, directory, file_name, file_name_no_suffix = load_float32_binary_file('/Users/metzler/Work/Python/SPECT_DL/TrainData/Phantoms/0.phantom')
        #data_phantom = np.reshape(data_phantom, (250, 250))
        #plt.imshow(data_phantom)
        #plt.show()

        # write file
        out_name = output_directory + 'Histoimages/' + file_name_no_suffix + '.histo'
        #print(f"Writing to {out_name}.")
        the_histo_image.tofile(out_name)

        # get the FBP
        theta = -np.linspace(0., 360., num_views, endpoint=False)
        reconstructed_image = iradon(data.T, theta=theta, filter_name='hann')
        #plt.imshow(reconstructed_image)
        #plt.show()


        out_name = output_directory + 'FBP/' + file_name_no_suffix + '.fbp'
        reconstructed_image.tofile(out_name)
