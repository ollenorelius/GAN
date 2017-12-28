import tensorflow as tf
import params as p
import numpy as np
import os
import re


def read_image_folders(folders):
    """Reads folders containing classes of images.
    Args:
       folders: list of folder names each containing a class.

    Returns:

        filenames: List with all filenames in file image_list_file.
        labels: matrix of one-hot vectors. [images, classes]
    """

    if type(folders) == str:
        folders = [folders]

    labels = []

    filename_list = [] # This is a list built from the names found in list.txt

    filenames = [] #This is all of the files in the folder
    print(folders)
    for i, folder in enumerate(folders):
        filenames_unfiltered = os.listdir(folder)
        for unf in filenames_unfiltered:
            if re.search('\.jpg\Z', unf) != None:
                filenames.append(folder + '/' + unf)
                classes = np.zeros(shape=(1, len(folders) + 1))
                classes[0,i] = 1
                labels.append(classes)

    return filenames, labels


def read_images_from_disk(filename):
    """Consumes a list of filenames, loads images from disc,
        and applies random transformations.
    Args:
      filename: An 1D string tensor.
    Returns:
      One tensor: the decoded images.
    """

    file_contents = tf.read_file(filename)
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.image.resize_images(image, [256,256])
    tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=False, name=None)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.1)
    tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=False, name=None)
    return image

def get_batch(size,folder):
    image_list, class_list = read_image_folders(folder)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    classes = tf.convert_to_tensor(np.asarray(class_list), dtype=tf.float32)

    tensor_slice = tf.train.slice_input_producer(
    [images, classes], shuffle=True)

    image = read_images_from_disk(tensor_slice[0])


    image_batch, class_batch = tf.train.batch([image,
                                              tensor_slice[1]], #classes
                                              batch_size=size)

    return image_batch, class_batch
