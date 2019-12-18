import numpy as np 
from PIL import ImageDraw
from PIL.ImageQt import ImageQt
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from PyQt5.QtGui import QIcon, QPixmap

def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)

class face_patch:
    def __init__(self, orig_image, file_name, encoding, location):
        top, right, bottom, left = location
        self.orig_image = orig_image
        self.orig_image_filename = file_name
        self.pil_patch = Image.fromarray(self.orig_image[top:bottom, left:right])
        self.image_with_box = Image.open(file_name).convert("RGBA")
        draw = ImageDraw.Draw(self.image_with_box)
        drawrect(draw, [(left, bottom), (right, top)], outline="red", width=8)
        del draw
        qim = ImageQt(self.image_with_box)
        self.pixmap_with_box = QPixmap.fromImage(qim)
        self.encoding = encoding
        self.location = location
        self.match_quality = 0.0



