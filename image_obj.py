import numpy as np 
from PIL.ImageQt import ImageQt
import face_recognition
from face_patch import face_patch
from PyQt5.QtGui import QIcon, QPixmap
class image_obj:
    def __init__(self, image_file_name):
        self.image_file_name = image_file_name
        self.face_rec_image = face_recognition.load_image_file(self.image_file_name)
        self.pixmap = QPixmap(self.image_file_name)
        self.face_patches = []
    
    def get_faces(self):
        face_locations = face_recognition.face_locations(self.face_rec_image, number_of_times_to_upsample=0, model="hog")
        face_encodings = face_recognition.face_encodings(self.face_rec_image, face_locations, num_jitters=10)
        for i,loc in enumerate(face_locations):
            patch_obj = face_patch(self.face_rec_image, self.image_file_name, face_encodings[i], loc)
            self.face_patches.append(patch_obj)