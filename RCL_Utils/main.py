import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.transform import AffineTransform
from skimage.transform import warp
from skimage.transform import rotate
import os
import shutil

def make_background(image_name):
    image = image_name #' '
    #print(image)
    file_without_extension = image.split('.')[0]
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    #print(image)
    trans_mask = image[:, :, 3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(file_without_extension + '.jpeg', new_img)

def resize(image_name):
    image = Image.open(image_name)
    image_resized = image.resize((28, 28), Image.ANTIALIAS)
    os.remove(image_name)
    image_resized_grayscale = ImageOps.grayscale(image_resized)
    image_resized_grayscale.save(image_name, 'JPEG', quality=100)

def shift_image(image_name):
    image = image_name #' '
    img = cv2.imread(image)
    file_without_extension = image.split('.')[0]
    arr_translation = [[15, -15], [-15, 15], [-15, -15],
                       [15, 15]]
    arr_caption = ['15-15', '-1515', '-15-15', '1515']
    for i in range(4):
        transform = AffineTransform(translation=tuple(arr_translation[i]))
        warp_image = warp(img, transform, mode="wrap")
        img_convert = cv2.convertScaleAbs(warp_image,alpha=(255.0))
        cv2.imwrite(file_without_extension + arr_caption[i] + '.jpeg', img_convert)


def rotate_image(image_name):
    image = image_name #' '
    img = Image.open(image)
    file_without_extension = image.split('.')[0]
    angles = np.ndarray((2,), buffer=np.array([-13, 13]), dtype=int)
    for angle in angles:
        transformed_image = rotate(np.array(img), angle, cval=255, preserve_range=True).astype(np.uint8)
        cv2.imwrite(file_without_extension + str(angle) + '.jpeg', transformed_image)


def balancing(root_path):
    arr_len_files = []
    for letter_number in range(1, 34):
        name_path = root_path+str(letter_number)+'/'
        files=os.listdir(name_path)
        arr_len_files.append(len(files))

    min_value=min(arr_len_files)
    for letter_number in range(1, 34):
        folder = root_path+str(letter_number)
        arr = []
        for the_file in os.listdir(folder):
            arr.append(folder + '/' + the_file)
        d = 0
        k = len(arr)
        for i in arr:
            os.remove(i)
            d += 1
            if d == k - min_value:
                break


def transform(image):
    #make_background(image)
    #shift_image(image)
    #rotate_image(image)
    resize(image)
    #os.remove(image)

def replace(image):
    make_background(image)
    os.remove(image)

def transform_images_of_letter(folder):
    for filename in os.listdir(folder):
        name = "/".join([folder, filename])
        print("transforming ", name)
        transform(name)

def replace_images_of_letter(folder):
    for filename in os.listdir(folder):
        name = "/".join([folder, filename])
        print("replacing ", name)
        replace(name)

def multiply_letter(letter):
    #letter = str(letter)
    letter_folder = "C:/Users/spess/PycharmProjects/RCL_Dataset_Improver/training/" + str(letter)
    #replace_images_of_letter(letter_folder)
    transform_images_of_letter(letter_folder)

def multiply_all():
    for letter in range(1, 34):
        multiply_letter(letter)

def move_to_test_group():
    for letter in range(1, 34):
        letter_folder = "C:/Users/spess/PycharmProjects/RCL_Dataset_Improver/training/" + str(letter)
        counter = 0
        for image in os.listdir(letter_folder):
            image_full = letter_folder + "/" + image
            print("moving ", image_full)
            shutil.move(image_full, "C:/Users/spess/PycharmProjects/RCL_Dataset_Improver/testing/" + str(letter))
            counter = counter + 1
            if counter == 722:
                break

multiply_all()

#balancing("C:/Users/spess/PycharmProjects/RCL_Dataset_Improver/Cyrillic/")

#move_to_test_group()