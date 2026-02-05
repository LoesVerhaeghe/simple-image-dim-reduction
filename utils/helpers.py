import pandas as pd
from os import listdir
from PIL import Image as PImage
import numpy as np

def extract_images(path_to_folders, image_type='all', magnification=10):
    """
    Extract images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        image_type (str): The type of images to extract: 'all', 'old', or 'new', train or test. Default is 'all'.
        old refers to old microscope in the lab, new refers to new microscope
        magnification: type of magnification (10 or 40)

    Returns:
        all_images (list): A list of all extracted images.
    """
    target_folder_microscope_type='2024-01-26'
    target_folder_dataset_type='2024-06-26'
    image_folders = sorted(listdir(path_to_folders)) 

    # Initialize lists for images and labels
    all_images = []

    # Define the condition for folder selection based on image_type
    if image_type == 'all':
        selected_folders = image_folders
    elif image_type == 'old':
        selected_folders = [folder for folder in image_folders if folder < target_folder_microscope_type]
    elif image_type == 'new':
        selected_folders = [folder for folder in image_folders if folder >= target_folder_microscope_type]
    elif image_type == 'train':
        selected_folders = [folder for folder in image_folders if folder <= target_folder_dataset_type]
    elif image_type == 'test':
        selected_folders = [folder for folder in image_folders if folder > target_folder_dataset_type]
    else:
        raise ValueError("Invalid image_type. Choose from 'all', 'old', or 'new' or 'train' or 'test'.")

    selected_folders = sorted(selected_folders)

    # Save all images and labels from the selected folders
    for folder in selected_folders:
        path_to_image = f"{path_to_folders}/{folder}/basin5/{magnification}x"
        images_list = sorted(listdir(path_to_image))
        for image in images_list:
            img = PImage.open(f"{path_to_image}/{image}")  # open in RGB color space
            all_images.append(img)

    return all_images

def extract_images_and_labels(path_to_images, labels_dataframe, image_type='all'):
    """
    Extract images and labels from the specified folder.

    Parameters:
        path_to_images (str): The base folder containing subfolders with images.
        labels_dataframe: pandas dataframe with labels saved
        image_type (str): The type of images to extract: 'all', 'old', or 'new'. Default is 'all'.

    Returns:
        all_images (list): A list of all extracted images.
        image_labels (numpy array): A numpy array of the corresponding labels.
    """
    target_folder_microscope_type='2024-01-26'
    target_folder_dataset_type='2024-06-26'
    image_folders = sorted(listdir(path_to_images)) 

    # Initialize lists for images and labels
    all_images = []
    image_labels = []

    # Define the condition for folder selection based on image_type
    if image_type == 'all':
        selected_folders = image_folders
    elif image_type == 'old':
        selected_folders = [folder for folder in image_folders if folder < target_folder_microscope_type]
    elif image_type == 'new':
        selected_folders = [folder for folder in image_folders if folder >= target_folder_microscope_type]
    elif image_type == 'train':
        selected_folders = [folder for folder in image_folders if folder <= target_folder_dataset_type]
    elif image_type == 'test':
        selected_folders = [folder for folder in image_folders if folder > target_folder_dataset_type]
    else:
        raise ValueError("Invalid image_type. Choose from 'all', 'old', or 'new' or 'train' or 'test'.")

    selected_folders = sorted(selected_folders)
      
    # Save all images and labels from the selected folders
    for folder in selected_folders:
        if folder not in labels_dataframe.index:   # <- skip missing label folders
            continue
        path = f"{path_to_images}/{folder}/basin5/10x"
        images_list = sorted(listdir(path))
        for image in images_list:
            img = PImage.open(f"{path}/{image}")  # open in RGB color space
            all_images.append(img)
            image_labels.append(labels_dataframe.loc[folder])

    # Convert image labels to numpy array
    image_labels = np.array(image_labels)

    return all_images, image_labels

def extract_images_and_labels_zurich(path_to_folders, labels_dataframe, start_folder, end_folder):
    """
    Extract paths from all the images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        start_folder: start date from which images need to be extracted
        end_folder: end date until which images need to be extracted

    Returns:
        all_images (list): A list of all extracted images.
    """
    image_folders = sorted(listdir(path_to_folders)) 

    all_img_paths=[]
    all_labels=[]

    # Select the images from start until end date
    selected_folders = [folder for folder in image_folders if start_folder <= folder <= end_folder]
    selected_folders = sorted(selected_folders)

    # Save all paths from the selected folders
    for folder in selected_folders:
        subfolders=sorted(listdir(f"{path_to_folders}/{folder}"))
        for subfolder in subfolders:
            path_to_image = f"{path_to_folders}/{folder}/{subfolder}"
            images_list = listdir(path_to_image)
            for image in images_list:
                img_path = f"{path_to_image}/{image}"
                all_img_paths.append(img_path)
                all_labels.append(labels_dataframe.loc[folder])

    # Convert image labels to numpy array
    all_labels = np.array(all_labels)

    return all_img_paths, all_labels



def extract_images_and_labels_pantarein(path_to_folders, labels_dataframe, start_folder, end_folder):
    """
    Extract paths from all the images from the specified folder.

    Parameters:
        base_folder (str): The base folder containing subfolders with images.
        start_folder: start date from which images need to be extracted
        end_folder: end date until which images need to be extracted

    Returns:
        all_images (list): A list of all extracted images.
    """
    image_folders = sorted(listdir(path_to_folders)) 

    all_img_paths=[]
    all_labels=[]

    # Select the images from start until end date
    selected_folders = [folder for folder in image_folders if start_folder <= folder <= end_folder]
    selected_folders = sorted(selected_folders)

    # Save all paths from the selected folders
    for folder in selected_folders:
        folder_path = f"{path_to_folders}/{folder}"
        images_list = listdir(folder_path)
        for image in images_list:
            img_path = f"{folder_path}/{image}"
            all_img_paths.append(img_path)
            all_labels.append(labels_dataframe.loc[folder])

    # Convert image labels to numpy array
    all_labels = np.array(all_labels)

    return all_img_paths, all_labels

