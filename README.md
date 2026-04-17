# Sistema de Reconocimiento Facial

Este proyecto es una herramienta de reconocimiento facial implementada originalmente en Python utilizando OpenCV y LBPH, y ahora adaptada para funcionar directamente en el navegador mediante **face-api.js**.

## 🚀 Características
- **Detección Facial en Tiempo Real**: Utiliza la cámara del dispositivo para detectar rostros.
- **Implementación Web**: Ejecutable desde cualquier navegador sin necesidad de instalar dependencias de Python.
- **Despliegue Automático**: Integrado con GitHub Actions para desplegar automáticamente en GitHub Pages.

## 🛠️ Tecnologías Utilizadas
- **Frontend**: HTML5, CSS3, JavaScript.
- **Librerías**: [face-api.js](https://github.com/justadudewhohacks/face-api.js/) (basada en TensorFlow.js).
- **Backend (Original)**: Python, OpenCV, LBPH.
- **CI/CD**: GitHub Actions.

## 📦 Instalación y Ejecución (Versión Web)
No es necesario instalar nada. Simplemente accede a la URL de GitHub Pages proporcionada en el repositorio.

## 🐍 Ejecución de la Versión Python (Legado)
Si deseas ejecutar la versión original en tu máquina local:
1. Instala las dependencias: `pip install -r requeriments.txt`
2. Recolecta datos: `python recolectarGrayScale.py`
3. Entrena el modelo: `python entrenandoRF.py`
4. Ejecuta el reconocimiento: `python ReconocimientoGrayScale.py`

## 📄 Licencia
Este proyecto es de código abierto.
