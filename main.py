from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
import cv2
import numpy as np
import time
import os 


class VentanaPrincipal(QMainWindow): 
    def __init__(self): 
        super().__init__()
        uic.loadUi('gui.ui', self)
        self.pushButton_cargar_img.clicked.connect(self.seleccionar_img)
        self.img_cargada = False
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.img_cargada:
            self.ajustar_img2label(self.imgpixmap_original, self.label_img_original, 2, 2)


    def seleccionar_img(self):
        archivo = QFileDialog()
        archivo.setWindowTitle("Seleccionar imagen")
        archivo.setFileMode(QFileDialog.ExistingFile)
        if archivo.exec_():
            ruta = archivo.selectedFiles()
            ruta = ruta[0]
            self.mostrar_img(ruta)
    
    def mostrar_img(self, ruta):
        ruta_absoluta = os.path.abspath(ruta)
        ruta_normalizada = os.path.normpath(ruta_absoluta)
        self.imgpixmap_original = QPixmap(ruta_normalizada)
        self.ajustar_img2label(self.imgpixmap_original, self.label_img_original, 0, 0)
        self.mostrar_datos_label(self.label_img_original_datos, self.imgpixmap_original.width(), self.imgpixmap_original.height())
        self.img_cargada = True
    
    def mostrar_datos_label(self, label, ancho, alto):
        label.setText(f'Ancho: {ancho}px ~ Alto: {alto}px')

    def ajustar_img2label(self, imgpixmap, label, margeny, margenx):
        if imgpixmap.height() > imgpixmap.width():
            imgpixmap = imgpixmap.scaledToHeight(label.height()-margeny)
        else:
            imgpixmap = imgpixmap.scaledToWidth(label.width()-margenx)
        label.setPixmap(imgpixmap)
    
    def cambio_tamanio(self):
        self.ajustar_img2label(self.imgpixmap_original, self.label_img_original, 2, 2)



app = QApplication([])
ventana = VentanaPrincipal()
ventana.show()
app.exec()