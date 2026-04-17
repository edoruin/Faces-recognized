# Sistema de Reconocimiento Facial con OpenCV

Este proyecto implementa un sistema de reconocimiento facial utilizando Python y la librería OpenCV, específicamente empleando el algoritmo **LBPH (Local Binary Patterns Histograms)**.

## 🚀 Funcionamiento
El sistema se divide en tres etapas principales:
1. **Recolección de datos**: Captura imágenes del rostro del usuario para crear una base de datos.
2. **Entrenamiento**: Procesa las imágenes recolectadas para generar un modelo matemático (`.xml`).
3. **Reconocimiento**: Utiliza la cámara en tiempo real para identificar rostros basándose en el modelo entrenado.

## 🛠️ Requisitos
Asegúrate de tener instalado Python y las siguientes dependencias:
- `opencv-contrib-python` (Necesario para el módulo de reconocimiento facial LBPH)
- `opencv-python`
- `imutils`
- `numpy`

Puedes instalar todas las dependencias ejecutando:
```bash
pip install -r requeriments.txt
```

## 📦 Guía de Uso

### 1. Recolección de Rostros
Ejecuta el script para capturar imágenes de una persona nueva:
```bash
python recolectarGrayScale.py
```
Sigue las instrucciones en la terminal, ingresa el nombre de la persona y mantente frente a la cámara hasta que se capturen las 50 imágenes requeridas.

### 2. Entrenamiento del Modelo
Una vez recolectados los datos, entrena el modelo de reconocimiento:
```bash
python entrenandoRF.py
```
Este script procesará las imágenes de la carpeta `data/` y guardará el modelo entrenado en `LBPH/modeloLBPHFace_.xml`.

### 3. Reconocimiento en Tiempo Real
Para iniciar el reconocimiento facial mediante la cámara:
```bash
python ReconocimientoGrayScale.py
```
El programa detectará los rostros y mostrará el nombre de la persona si el nivel de confianza es adecuado. Presiona la tecla `ESC` para salir.

## 📂 Estructura del Proyecto
- `recolectarGrayScale.py`: Captura de imágenes.
- `entrenandoRF.py`: Entrenamiento del modelo LBPH.
- `ReconocimientoGrayScale.py`: Ejecución del reconocimiento facial.
- `LBPH/`: Carpeta donde se almacena el modelo entrenado.
- `data/`: Carpeta donde se guardan las imágenes de los rostros recolectados.
