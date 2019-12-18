import numpy as np 
from PIL.ImageQt import ImageQt
import face_recognition
from image_obj import image_obj
from os import listdir
from os.path import isfile, join, splitext, isdir

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QLabel
import sys
import multiprocessing


def get_file_names(path):
    file_names = []
    image_extensions = [".jpeg", ".jpg", ".png"]
    for image_file_name in listdir(path):
        print("File name:", image_file_name)
        file_path = join(path,image_file_name)
        if isdir(file_path):
            temp_file_names = get_file_names(file_path)
            file_names.extend(temp_file_names)
        elif isfile(file_path):
            filename, extension = splitext(image_file_name)
            if extension in image_extensions:
                file_names.append(join(path, image_file_name))
    return file_names

def get_encodings(file_list):
    num_files = len(file_list)
    all_encodings = get_enc((0,num_files), file_list)
    return all_encodings
    '''
    num_p = 4
    starts = []
    ends = []
    for i in range(num_p):
        starts.append(int(i*num_files/num_p))
        ends.append(int((i+1)*num_files/num_p))
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    jobs = []
    for i in range(num_p):
        process = multiprocessing.Process(target=get_enc, 
                                          args=(i, (starts[i], ends[i]), file_list, return_dict))
        jobs.append(process)

    # Start the processes (i.e. calculate the random number lists)      
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for i,j in enumerate(jobs):
        print("waiting for %i of %i", i, len(jobs))
        j.join()

    all_encodings = []
    for val in return_dict.values():
        print(val)
        all_encodings.append(val)

    print (len(all_encodings))
    return all_encodings
    '''


def get_enc(bounds, file_paths):
    all_encodings = []
    num_files = bounds[1] - bounds[0]
    counter = 0
    for i in range(bounds[0], bounds[1]):
        print("On file " + str(counter) + " of " + str(num_files))
        counter += 1
        face_rec_image = face_recognition.load_image_file(file_paths[i])
        face_locations = face_recognition.face_locations(face_rec_image, number_of_times_to_upsample=0, model="cnn")
        face_encodings = face_recognition.face_encodings(face_rec_image, face_locations, num_jitters=10)
        all_encodings.extend(face_encodings)
    return all_encodings
def get_numpy_array(encodings):
    array = np.zeros(shape=(len(encodings), len(encodings[0])))
    for i,enc in enumerate(encodings):
        print(array[i])
        print(np.array(enc))
        array[i] = np.array(enc)
    return array

path = "/home/vader/digi_forensics/not_poi"
file_names = get_file_names(path)
all_encodings = get_encodings(file_names)
array = get_numpy_array(all_encodings)
print("Saving File")
np.save("/home/vader/digi_forensics/not_poi_array", array)
print("File Saved")
