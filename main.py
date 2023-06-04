import random

from pictures_generator import blend_pictures
import os
import imageio
from moviepy.editor import *
import itertools

PROJECT_FOLDER = os.getcwd()
CUP_FOLDER = os.path.join(PROJECT_FOLDER, 'cup')
FIGURE_FOLDER = os.path.join(PROJECT_FOLDER, 'figure')
MASK_FOLDER = os.path.join(PROJECT_FOLDER, 'mask')
OUTPUT_FOLDER = os.path.join(PROJECT_FOLDER, 'output')
FADE_FOLDER = os.path.join(PROJECT_FOLDER, 'test')


def get_numeric_value(filename):
    return int(os.path.splitext(os.path.basename(filename))[0].split("_")[0])


def generate_all_pictures(scale=0.5):
    """
    Blend all pictures in cup folder.
    :param scale: float between 1 and 0, decides the blend level in the mask area
    :return:
    """
    # Get the list of image files in the cup folder
    cup_files = sorted(
        [os.path.join(CUP_FOLDER, file) for file in os.listdir(CUP_FOLDER) if file.endswith(('.jpg', '.jpeg', '.png'))],
        key=get_numeric_value)

    # Iterate over the cup images
    for i, cup_file in enumerate(cup_files):
        # Get the corresponding figure and mask file paths
        figure_file = os.path.join(FIGURE_FOLDER, f"{i + 1}.jpg")
        mask_file = os.path.join(MASK_FOLDER, f"{i + 1}.jpg")

        # Create the output file path
        output_file = os.path.join(OUTPUT_FOLDER, f"{i + 1}_{scale}.jpg")

        # Call the blend_pictures function
        if not os.path.exists(output_file):
            blend_pictures(cup_file, figure_file, mask_file, output_file, scale)


def generate_video():
    """
    Generate output video.
    :return:
    """
    # # Get the list of image files in the folder
    image_files = sorted([os.path.join(OUTPUT_FOLDER, file) for file in os.listdir(OUTPUT_FOLDER) if
                          file.endswith(('.jpg', '.jpeg', '.png'))],
                         key=get_numeric_value)

    fade_files = sorted([os.path.join(FADE_FOLDER, file) for file in os.listdir(FADE_FOLDER) if
                         file.endswith(('.jpg', '.jpeg', '.png'))],
                        key=get_numeric_value)

    image_files = fade_files + image_files
    # Create an empty list to store the image frames
    frames = []

    # Read each image and append it to the frames list
    for image_file in image_files:
        image = imageio.imread(image_file)
        frames.append(image)

    # Create a video writer object
    video_writer = imageio.get_writer('output_video.mp4', fps=3)  # Set the desired frames per second (fps)

    # Write each image frame to the video twice
    for j in range(2):
        for frame in itertools.chain(frames, reversed(frames)):
            video_writer.append_data(frame)

    # Close the video writer
    video_writer.close()


if __name__ == '__main__':
    generate_all_pictures()
    generate_video()
