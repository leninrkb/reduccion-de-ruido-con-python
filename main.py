from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
import cv2
from matplotlib.animation import ImageMagickWriter
import numpy as np
import time
import os
from scipy.ndimage import median_filter


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("gui.ui", self)
        self.pushButton_cargar_img.clicked.connect(self.seleccionar_img)
        self.pushButton_aplicar_ruido.clicked.connect(self.aplicar_ruido)
        self.pushButton_python_aplicar.clicked.connect(self.aplicar_python)
        self.pushButton_python_descargar.clicked.connect(self.descargar_python)
        self.pushButton_descargar_img_ruido.clicked.connect(self.descargar_ruido)
        self.radioButton_img_original.setChecked(False)
        self.radioButton_img_ruido.setChecked(False)
        self.img_cargada = False
        self.img_ruido_cargada = False
        self.img_python_cargada = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.img_cargada:
            self.ajustar_img2label(self.pixmap_original, self.label_img_original)
        if self.img_ruido_cargada:
            self.ajustar_img2label(self.pixmap_ruido, self.label_img_ruido)
        if self.img_python_cargada:
            self.ajustar_img2label(self.pixmap_python_resultado, self.label_python_img)

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
        self.pixmap_original = QPixmap(ruta_normalizada)
        self.ajustar_img2label(self.pixmap_original, self.label_img_original)
        self.mostrar_datos_label(self.label_img_original_datos, self.pixmap_original)
        self.img_cargada = True
        self.radioButton_img_original.setChecked(True)
        self.img_original = cv2.imread(ruta_normalizada)
        self.img_original = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
        self.img_trabajo = self.img_original

    def mostrar_datos_label(self, label=None, pixmap=None, procesando=False):
        if not procesando:
            label.setText(f"Ancho: {pixmap.width()}px ~ Alto: {pixmap.height()}px")
        else:
            label.setText("procesando imagen...")
        QApplication.processEvents()

    def ajustar_img2label(self, imgpixmap, label):
        imgpixmap = imgpixmap.scaled(label.size(), aspectRatioMode=True)
        label.setPixmap(imgpixmap)

    def aplicar_ruido(self):
        ruido = np.zeros(self.img_original.shape, dtype=np.uint8)
        media = self.doubleSpinBox_media.value()
        desviacion = self.doubleSpinBox_desviacion.value()
        cv2.randn(ruido, media, desviacion)
        self.img_ruido = cv2.add(self.img_original, ruido)
        self.mostrar_img_ruido()
        self.img_ruido_cargada = True
        self.cambio_img_trabajo()

    def mostrar_img_ruido(self):
        self.pixmap_ruido = self.imgcv2pixmap(self.img_ruido)
        self.ajustar_img2label(self.pixmap_ruido, self.label_img_ruido)
        self.mostrar_datos_label(self.label_datos_img_ruido, self.pixmap_ruido)

    def imgcv2pixmap(self, img):
        altoimg, anchoimg, channels = img.shape
        bytes_linea = channels * anchoimg
        q_image = QImage(
            img.data.tobytes(), anchoimg, altoimg, bytes_linea, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    def cambio_img_trabajo(self):
        self.radioButton_img_original.setChecked(
            not self.radioButton_img_original.isChecked()
        )
        self.radioButton_img_ruido.setChecked(
            not self.radioButton_img_ruido.isChecked()
        )
        self.img_trabajo = (
            self.img_original
            if self.radioButton_img_original.isChecked()
            else self.img_ruido
        )

    def aplicar_python(self):
        self.mostrar_datos_label(label=self.label_python_datos_img, procesando=True)
        if self.radioButton_python_espacio.isChecked():
            ancho = self.spinBox_python_espacio_ancho.value()
            alto = self.spinBox_python_espacio_alto.value()
            r, g, b = cv2.split(self.img_trabajo)
            r = median_filter(r, size=(ancho, alto))
            g = median_filter(g, size=(ancho, alto))
            b = median_filter(b, size=(ancho, alto))
            self.img_python_resultado = cv2.merge([r, g, b])
            self.pixmap_python_resultado = self.imgcv2pixmap(self.img_python_resultado)
            self.ajustar_img2label(self.pixmap_python_resultado, self.label_python_img)
            self.mostrar_datos_label(
                self.label_python_datos_img, self.pixmap_python_resultado
            )
            self.img_python_espacio_cargada = True

        elif self.radioButton_python_frecuencia.isChecked():
            radio = self.spinBox_python_frecuencia_radio.value()
            r, g, b = cv2.split(self.img_trabajo)
            r = self.python_filtro_altos(radio, r)
            g = self.python_filtro_altos(radio, g)
            b = self.python_filtro_altos(radio, b)
            self.img_python_resultado = cv2.merge([r,g,b])
            self.pixmap_python_resultado = self.imgcv2pixmap(self.img_python_resultado)
            self.ajustar_img2label(self.pixmap_python_resultado, self.label_python_img)

    def python_filtro_altos(self, radio, img):
        gray = img 
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        rows, cols = gray.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols), np.uint8)
        mask[crow - radio : crow + radio, ccol - radio : ccol + radio] = 0
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        image_filtered = np.abs(np.fft.ifft2(f_ishift))
        resultado = cv2.normalize(
            image_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        return resultado

    def descargar_img(self, img, nombre):
        directorio = QFileDialog.getExistingDirectory(ventana, "Seleccionar directorio")
        if directorio:
            idtiempo = int(time.time())
            titulo = f"/{idtiempo}_{nombre}_img.jpg"
            archivo = directorio + titulo
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(archivo, img)

    def descargar_python(self):
        self.descargar_img(
            self.img_python_resultado,
            "python_espacio"
            if self.radioButton_python_espacio.isChecked()
            else "python_frecuencia",
        )

    def descargar_ruido(self):
        self.descargar_img(self.img_ruido, "ruido_agregado")


app = QApplication([])
ventana = VentanaPrincipal()
ventana.show()
app.exec()
