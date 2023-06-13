from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow, QButtonGroup
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
        uic.loadUi("gui2.ui", self)
        self.pushButton_cargar_img.clicked.connect(self.seleccionar_img)
        self.pushButton_aplicar_ruido.clicked.connect(self.aplicar_ruido)
        self.pushButton_aplicar_filtro.clicked.connect(self.aplicar_filtro)
        self.pushButton_descargar_resultado.clicked.connect(self.descargar_resultado)
        self.pushButton_descargar_img_ruido.clicked.connect(self.descargar_ruido)
        # 
        self.img_original_activa = False
        # 
        # 
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
        # 
        self.pixmap_original = QPixmap(ruta_normalizada)
        self.ajustar_img2label(self.pixmap_original, self.label_img_original)
        self.mostrar_datos_label(self.label_img_original_datos, self.pixmap_original, mns='')
        # 
        self.img_original = cv2.imread(ruta_normalizada)
        self.img_original = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
        self.img_original_activa = True
        self.cambio_img_trabajo()
        # 
        self.img_cargada = True
        self.pushButton_aplicar_ruido.setEnabled(True)

    def mostrar_datos_label(self, label=None, pixmap=None, procesando=False, mns=None):
        if not procesando:
            label.setText(f"{mns} Ancho: {pixmap.width()}px ~ Alto: {pixmap.height()}px")
        elif not mns == None:
            label.setText(mns)
        else:
            label.setText("procesando imagen...")
        QApplication.processEvents()

    def ajustar_img2label(self, imgpixmap, label):
        imgpixmap = imgpixmap.scaled(label.size(), aspectRatioMode=True)
        label.setPixmap(imgpixmap)

    def aplicar_ruido(self):
        ruido = np.zeros(self.img_original.shape, dtype=np.uint8)
        media = self.spinBox_media.value()
        desviacion = self.spinBox_desviacion.value()
        cv2.randn(ruido, media, desviacion)
        self.img_ruido = cv2.add(self.img_original, ruido)
        self.mostrar_img_ruido()
        self.img_ruido_cargada = True
        self.img_original_activa = False
        self.cambio_img_trabajo()

    def mostrar_img_ruido(self):
        self.pixmap_ruido = self.imgcv2pixmap(self.img_ruido)
        self.ajustar_img2label(self.pixmap_ruido, self.label_img_ruido)
        self.mostrar_datos_label(self.label_datos_img_ruido, self.pixmap_ruido, mns='ruido')

    def imgcv2pixmap(self, img):
        altoimg, anchoimg, channels = img.shape
        bytes_linea = channels * anchoimg
        q_image = QImage(
            img.data.tobytes(), anchoimg, altoimg, bytes_linea, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    def cambio_img_trabajo(self):
        self.img_trabajo = (
            self.img_original
            if self.img_original_activa
            else self.img_ruido
        )

    def aplicar_filtro(self):
        if self.radioButton_manual.isChecked():
            self.aplicar_manual()
        elif self.radioButton_python.isChecked():
            self.aplicar_python()
        
    def aplicar_manual(self):
        print()

    def aplicar_python(self):
        self.img_python_cargada = False
        self.mostrar_datos_label(label=self.label_python_datos_img, procesando=True)
        if self.radioButton_espacio.isChecked():
            ancho = self.spinBox_filtro_x.value()
            alto = self.spinBox_filtro_y.value()
            r, g, b = cv2.split(self.img_trabajo)
            r = median_filter(r, size=(ancho, alto))
            g = median_filter(g, size=(ancho, alto))
            b = median_filter(b, size=(ancho, alto))
        elif self.radioButton_frecuencia.isChecked():
            radio = self.spinBox_filtro_frecuencia.value()
            radio = self.verificar_radio(radio,self.img_trabajo,self.spinBox_filtro_frecuencia)
            r, g, b = cv2.split(self.img_trabajo)
            if self.radioButton_altos.isChecked():
                r = self.python_filtro_altos(r, radio)
                g = self.python_filtro_altos(g, radio)
                b = self.python_filtro_altos(b, radio)
            elif self.radioButton_bajos.isChecked():
                r = self.python_filtro_bajos(r, radio)
                g = self.python_filtro_bajos(g, radio)
                b = self.python_filtro_bajos(b, radio)
        self.img_python_resultado = cv2.merge([r, g, b])
        self.pixmap_python_resultado = self.imgcv2pixmap(self.img_python_resultado)
        self.ajustar_img2label(self.pixmap_python_resultado, self.label_python_img)
        self.mostrar_datos_label(self.label_python_datos_img, self.pixmap_python_resultado, mns='Python')
        self.img_python_cargada = True
        self.pushButton_descargar_resultado.setEnabled(True)

    def python_filtro_altos(self, img, radio):
        fourier = np.fft.fft2(img)
        fdesplazado = np.fft.fftshift(fourier)
        filas, columnas = img.shape
        centro_filas, centro_columnas = filas // 2, columnas // 2
        mascara = np.zeros((filas, columnas), np.uint8)
        mascara[centro_filas - radio:centro_filas + radio, centro_columnas - radio:centro_columnas + radio] = 1
        filtrado = fdesplazado * mascara
        inversa_desplazado = np.fft.ifftshift(filtrado)
        img = np.abs(np.fft.ifft2(inversa_desplazado))
        return img.astype(np.uint8)
    
    def python_filtro_bajos(self, img, radio):
        fourier = np.fft.fft2(img)
        fdesplazado = np.fft.fftshift(fourier)
        filas, columnas = img.shape
        centro_filas, centro_columnas = filas // 2, columnas // 2
        mascara = np.ones((filas, columnas), np.uint8)
        mascara[centro_filas - radio:centro_filas + radio, centro_columnas - radio:centro_columnas + radio] = 0
        filtrado = fdesplazado * mascara
        inversa_desplazado = np.fft.ifftshift(filtrado)
        img = np.abs(np.fft.ifft2(inversa_desplazado))
        return img.astype(np.uint8)
    
    def verificar_radio(self, radio, img, objeto):
        x, y, _ = img.shape
        if x < y:
            x = y
        if radio*2 > x:
            x = x//2
            objeto.setValue(x)
            return x
        return radio
    
    def descargar_img(self, img, nombre):
        directorio = QFileDialog.getExistingDirectory(ventana, "Seleccionar directorio")
        if directorio:
            idtiempo = int(time.time())
            titulo = f"/{idtiempo}_{nombre}_img.jpg"
            archivo = directorio + titulo
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(archivo, img)

    def descargar_resultado(self):
        if self.radioButton_manual.isChecked():
            print()
        elif self.radioButton_python.isChecked():
            self.descargar_img(
                self.img_python_resultado,
                "python_espacio"
                if self.radioButton_espacio.isChecked()
                else "python_frecuencia_altos" 
                    if self.radioButton_altos.isChecked()
                    else "python_frecuencia_bajos",
            )

    def descargar_ruido(self):
        self.descargar_img(self.img_ruido, "ruido_agregado")


app = QApplication([])
ventana = VentanaPrincipal()
ventana.show()
app.exec()
