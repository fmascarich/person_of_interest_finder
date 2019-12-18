import numpy as np 
from PIL.ImageQt import ImageQt
import face_recognition
from image_obj import image_obj
from os import listdir, mkdir
from os.path import isfile, join, splitext, isdir
import os.path

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QLabel
import sys
from sklearn import svm, neighbors
from sklearn.svm import OneClassSVM

import shutil

class Ui(QtWidgets.QMainWindow):
    def __init__(self, q_app):
        super(Ui, self).__init__()
        uic.loadUi('main.ui', self)
        self.q_app = q_app
        # set to first page
        self.stackedWidget.setCurrentIndex(0)
        # setup callbacks
        self.button_browse_image_dir.clicked.connect(self.browse_image_dir)
        self.button_browse_poi_dir.clicked.connect(self.browse_poi_dir)
        self.button_start_search.clicked.connect(self.start_search_cb)
        self.button_yes.clicked.connect(self.yes_cb)
        self.button_no.clicked.connect(self.no_cb)
        self.button_browse_output_dir.clicked.connect(self.browse_output_dir)
        self.button_save_filtered_images.clicked.connect(self.save_filtered_images)
        self.button_save_filtered_images.setEnabled(False)

        # setup members
        self.image_dir_path = "/home/vader/digi_forensics/images"
        self.poi_dir_path = "/home/vader/digi_forensics/poi"
        self.not_poi_dir_path = "/home/vader/digi_forensics/not_poi"
        self.update_start_button()
        self.poi_img_objs = []
        self.other_img_objs = []
        self.filtered_out_patch_objs = []
        self.not_poi_enc_array = np.load("/home/vader/digi_forensics/not_poi_array.npy")
        self.current_eval_image_index = 0
        # show the widget
        self.show()

    def browse_output_dir(self):
        self.output_path = str(QFileDialog.getExistingDirectory(self, "Select Output Directory"))
        self.line_output_path.setText(self.output_path)
        if self.output_path == "":
            self.button_save_filtered_images.setEnabled(False)
        else:
            self.button_save_filtered_images.setEnabled(True)

    def save_filtered_images(self):
        output_path = join(self.output_path, "output_" + self.name_string.replace(" ", "_"))
        if not isdir(output_path):
            mkdir(output_path)
        for output_patch in self.filtered_out_patch_objs:
            file_name = os.path.basename(output_patch.orig_image_filename)
            dest_file_path = join(output_path, file_name)
            i = 0
            while isfile(dest_file_path):
                dest_file_path = dest_file_path+"_"+str(i)
                i+=1
            print("Saving : %s to %s", output_patch.orig_image_filename, dest_file_path)
            shutil.copy2(output_patch.orig_image_filename, dest_file_path)
        print("Exiting...")
        exit()

    def start_search_cb(self):
        self.name_string = str(self.line_person_name.text())
        # load all poi images
        poi_image_file_paths = self.get_file_names(self.poi_dir_path)
        self.stackedWidget.setCurrentIndex(1)
        num_poi_patches = 0
        self.poi_patches = []
        for i, poi_image_file_path in enumerate(poi_image_file_paths):
            self.update_progress_bar("Loading POI Image " + str(i + 1) + " of " + str(len(poi_image_file_paths)), i,  len(poi_image_file_paths))
            temp_img_obj = image_obj(poi_image_file_path)
            temp_img_obj.get_faces()
            self.poi_patches.extend(temp_img_obj.face_patches)
            self.poi_img_objs.append(temp_img_obj)

        other_image_file_paths = self.get_file_names(self.image_dir_path)
        self.other_image_patches = []
        for i, other_image_file_path in enumerate(other_image_file_paths):
            self.update_progress_bar("Loading Search Image " + str(i + 1) + " of " + str(len(other_image_file_paths)), i, len(other_image_file_paths))
            temp_img_obj = image_obj(other_image_file_path)
            temp_img_obj.get_faces()
            self.other_image_patches.extend(temp_img_obj.face_patches)
            self.other_img_objs.append(temp_img_obj)
        
        self.update_progress_bar("Training Classifier...", 0, 10)
        self.get_svm()
        self.update_predictions()

        while(not self.setup_eval_page(self.current_eval_image_index)):
            self.current_eval_image_index += 1
        self.stackedWidget.setCurrentIndex(2)

    def update_predictions(self):
        print("Updating Predictions")
        self.rebuild_enc_array()
        num_enc = self.all_enc_array.shape[0]
        num_feat = self.all_enc_array.shape[1]

        for i,other_patch in enumerate(self.other_image_patches):
            self.update_progress_bar("Classifying...", i, len(self.other_image_patches))
            encoding = np.array(other_patch.encoding)
            #matches = face_recognition.compare_faces(enc_list, encoding)
            #result = self.clf.predict(encoding.reshape(1, -1))
            dists, nearest_i = self.clf.kneighbors(encoding.reshape(1,-1), n_neighbors=1)
            print("RESULT : ", dists, nearest_i)
            nearest_i = nearest_i[0][0]
            dist = dists[0][0]
            print(nearest_i, dist)
            if nearest_i < len(self.poi_patches):
                self.other_image_patches[i].match_quality = 1/dist
                print("Found POI")
            else:
                self.other_image_patches[i].match_quality = -1.0
                print("Not POI")
            # if result[0] != -1:
            #     print ("Found not 1!")
            # #print("Num Votes for POI : ", votes[1])
            # self.other_image_patches[i].match_quality = result[0]

        self.other_image_patches.sort(key=lambda x: x.match_quality, reverse=True)

    def get_svm(self):
        self.rebuild_enc_array()

        self.clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
        self.clf.fit(self.all_enc_array, self.all_enc_labels)

        #self.clf = OneClassSVM(gamma='auto', kernel='rbf')
        #self.clf.fit(self.poi_enc_array)  

    def rebuild_enc_array(self):
        num_features = len(self.poi_patches[0].encoding)
        num_poi = len(self.poi_patches)
        num_not_poi = self.not_poi_enc_array.shape[0]

        self.poi_enc_array = np.zeros(shape=(num_poi, num_features))

        self.all_enc_array = np.zeros(shape=(num_poi + num_not_poi, num_features))
        self.all_enc_labels = []
        for i in range(num_poi):
            self.all_enc_array[i] = self.poi_patches[i].encoding
            self.poi_enc_array[i] = self.poi_patches[i].encoding
            self.all_enc_labels.append(1)

        for j in range(num_not_poi):
            self.all_enc_array[j+num_poi] = self.not_poi_enc_array[j]
            self.all_enc_labels.append(0)
        self.all_enc_labels = np.array(self.all_enc_labels)
        
    def setup_eval_page(self, image_index):
        if image_index >= len(self.other_image_patches):
            return False
        self.label_eval_title.setText("Is this " + self.name_string + "?")
        pixmap = self.other_image_patches[image_index].pixmap_with_box
        if pixmap.height() > 300:
            pixmap = pixmap.scaledToHeight(300)
        if pixmap.width() > 500:
            pixmap = pixmap.scaledToWidth(500)
        self.label_test_image.setPixmap(pixmap)
        if self.other_image_patches[image_index].match_quality > 0:
            self.label_class.setText("Match")
        else:
            self.label_class.setText("Not a Match")

        self.label_conf.setText("Quality : " + str(self.other_image_patches[image_index].match_quality))

        return True

    def yes_cb(self):
        new_poi_patch_obj = self.other_image_patches[0]
        self.poi_patches.append(new_poi_patch_obj)
        self.filtered_out_patch_objs.append(new_poi_patch_obj)
        del self.other_image_patches[0]
        self.get_svm()
        self.update_predictions()
        self.eval_next()

    def no_cb(self):
        new_not_poi_patch_obj = self.other_image_patches[0]
        np.append(self.not_poi_enc_array, new_not_poi_patch_obj.encoding)
        del self.other_image_patches[0]
        self.get_svm()
        self.update_predictions()
        self.eval_next()

    def eval_next(self):
        if  len(self.other_image_patches) <= 0:
            self.stackedWidget.setCurrentIndex(3)
            return
        while(not self.setup_eval_page(0) and len(self.other_image_patches) > 0):
            del self.other_image_patches[0]
            self.current_eval_image_index += 1
        if  len(self.other_image_patches) <= 0:
            self.stackedWidget.setCurrentIndex(3)
            return

    def get_file_names(self, path):
        file_names = []
        image_extensions = [".jpeg", ".jpg", ".png"]
        for image_file_name in listdir(path):
            filename, extension = splitext(image_file_name)
            if extension in image_extensions:
                file_names.append(join(path, image_file_name))
        return file_names

    def update_progress_bar(self, out_string, num, denom):
        self.label_image_proc.setText(out_string)
        ratio = float(num)/float(denom) * 100.0
        self.progress_image_proc.setValue(ratio)
        self.repaint()
        self.update()
        self.q_app.processEvents() 

    def browse_image_dir(self):
        self.image_dir_path = str(QFileDialog.getExistingDirectory(self, "Select Image Directory"))
        self.line_image_dir_path.setText(self.image_dir_path)
        self.update_start_button()
    
    def browse_poi_dir(self):
        self.poi_dir_path = str(QFileDialog.getExistingDirectory(self, "Select POI Directory"))
        self.line_poi_dir_path.setText(self.poi_dir_path)
        self.update_start_button()

    def update_start_button(self):
        self.name_string = str(self.line_person_name.text())
        if self.image_dir_path != "" and self.poi_dir_path != "" and self.image_dir_path != self.poi_dir_path and self.name_string != "":
            self.button_start_search.setEnabled(True)
        else:
            self.button_start_search.setEnabled(False)




app = QtWidgets.QApplication(sys.argv)
window = Ui(app)
app.exec_()