import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from concurrent.futures import ThreadPoolExecutor
import argparse

# Define a dictionary of supported image formats
SUPPORTED_IMAGE_FORMATS = {
    'jpg': 'JPEG',
    'png': 'PNG' ,
    'bmp': 'BMP'
}

def get_keys_by_value(d, v):
    return [k for k, val in d.items() if val == v]

def create_mask(json_file_path, output_mask_path, num_channels):
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Get the image width and height from the JSON data
    width = json_data['imageWidth']
    height = json_data['imageHeight']
    if num_channels==1:
    # Create a mask as a numpy array
        mask = np.zeros((height, width), dtype=np.uint8)

        background_vertices = []

        # Loop through each object in the JSON data
        for obj in json_data['shapes']:

            # Get the polygon vertices for this object
            vertices = obj['points']

            # Check if the shape is named "background"
            if obj['label'] == 'background':
                background_vertices.append(vertices)
                continue

            # Draw the polygon on the mask
            polygon = np.array(vertices, np.int32)
            polygon = polygon.reshape((-1,1,2))
            cv2.fillPoly(mask, [polygon], 255)

        # Fill the "background" shapes with black
        for vertices in background_vertices:
            polygon = np.array(vertices, np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon], 0)
    # Save the mask with the same format as the input image
    else:
        # Create a mask as a numpy array
        mask = np.zeros((height, width,3), dtype=np.uint8)

        background_vertices = []

        # Loop through each object in the JSON data
        for obj in json_data['shapes']:

            # Get the polygon vertices for this object
            vertices = obj['points']

            # Check if the shape is named "background"
            if obj['label'] == 'background':
                background_vertices.append(vertices)
                continue

            # Draw the polygon on the mask
            polygon = np.array(vertices, np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon], (255,255,255))

        # Fill the "background" shapes with black
        for vertices in background_vertices:
            polygon = np.array(vertices, np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [polygon], (0,0,0))
    mask_image = Image.fromarray(mask)
    mask_image.save(output_mask_path)


def convert_images_to_masks(input_folder, output_folder, num_channels, num_threads=4):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image file names in the input folder
    image_file_names = [f for f in os.listdir(input_folder) if f.lower().endswith(tuple(SUPPORTED_IMAGE_FORMATS.keys()))]
    total_images = len(image_file_names)

    # Use tqdm to show a progress bar
    with tqdm(total=total_images, desc='Converting images to masks') as pbar:
        num_saved_masks = 0

        # Create a thread pool with the specified number of threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Loop through all files in the input folder
            for filename in image_file_names:
                # Construct the input and output file paths
                image_file_path = os.path.join(input_folder, filename)
                json_file_path = os.path.join(input_folder, os.path.splitext(filename)[0] + '.json')

                # Get the input image format
                input_image_format = SUPPORTED_IMAGE_FORMATS[os.path.splitext(filename)[1][1:]]

                output_mask_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.' +get_keys_by_value(SUPPORTED_IMAGE_FORMATS, input_image_format)[0])

                # Submit the task to the thread pool
                future = executor.submit(create_mask, json_file_path, output_mask_path, num_channels)

                # Add a callback to the future to update the progress bar
                future.add_done_callback(lambda p: pbar.update(1))

                # Increment the number of saved masks
                num_saved_masks += 1

        # Print the number of saved masks and total number of images
        print(f'Saved {num_saved_masks} masks out of {total_images} images')


if __name__ == '__main__':
    # Get the input and output folder paths from the command line arguments

    parser = argparse.ArgumentParser(description='Convert images to masks')
    parser.add_argument('--input_folder', help='Path to input folder')
    parser.add_argument('--output_folder', help='Path to output folder')
    parser.add_argument('--num_channels', help='number of channels')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads')

    args = parser.parse_args()

    # Convert the images in the input folder to masks in the output folder using multiple threads
    convert_images_to_masks(args.input_folder, args.output_folder, args.num_channels, args.num_threads)