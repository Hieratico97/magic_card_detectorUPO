# 🎓 Magic Card Detector UPO - Proyecto Guiado de Visión por Computador

¡Bienvenido/a a este proyecto! Este repositorio no solo es una herramienta funcional para escanear y reconocer cartas de Magic: The Gathering (MTG), sino también un **proyecto de referencia didáctica** diseñado para aprender **Visión por Computador**.

Si alguna vez te has preguntado cómo tu teléfono o un escáner pueden detectar un documento, recortarlo en perspectiva, analizar si hay texto y qué dice, o saber si la imagen corresponde a un objeto en su base de datos... **este proyecto te enseña todo ese proceso paso a paso**.

---

## 🛠️ Herramientas Utilizadas

Este proyecto es un "pipeline" (una línea de montaje) de procesamiento de imágenes. Las principales tecnologías que forman parte de él son:

1. **[OpenCV](https://opencv.org/) (`cv2`)**: El corazón del proyecto. Es la librería de Visión por Computador más popular del mundo. Se utiliza para leer la imagen, convertirla a blanco y negro, detectar los bordes de la carta y corregir la perspectiva (enderezarla).
2. **[ImageHash](https://pypi.org/project/ImageHash/) (`imagehash`)**: Se encarga del "Perceptual Hashing". En lugar de comparar imágenes píxel a píxel (lo cual fallaría si hay cambios de luz, brillos o giros), genera una "huella digital" de la imagen y la compara con nuestra base de datos para ver a qué carta de MTG se parece más.
3. **[EasyOCR](https://github.com/JaidedAI/EasyOCR)**: Una herramienta de Reconocimiento Óptico de Caracteres (OCR, por sus siglas en inglés) impulsada por IA. Lee el texto impreso en la carta (como la edición o el idioma).
4. **[Shapely](https://pypi.org/project/shapely/) y Matemáticas Base (`numpy` / `scipy`)**: Nos ayudan con operaciones geométricas (analizar si un contorno es un rectángulo válido) y con las rotaciones de matrices (imágenes).
5. **[Flask](https://flask.palletsprojects.com/)**: El framework web utilizado para envolver todo nuestro cerebro de visión en una aplicación web interactiva.

---

## 🧠 ¿Cómo funciona el Pipeline de Visión? (Paso a Paso)

El código fuente principal está en `magic_card_detector.py`. Esto es lo que ocurre cuando subimos una foto:

### 1. Preprocesamiento de la Imagen (Mejora y Limpieza)
Antes de que el ordenador pueda "ver" la carta, tenemos que ayudarle.
- **Color a Escala de Grises**: Los ordenadores ven el color como 3 matrices (Rojo, Verde, Azul). Para detectar bordes, el color suele estorbar, así que transformamos la imagen a blanco y negro (`cv2.cvtColor`).
- **Ecualización del Histograma (CLAHE)**: A veces la foto es muy oscura o tiene brillos en la funda de la carta. Aplicamos `cv2.createCLAHE()` para nivelar las sombras y adaptar los contrastes locales, de forma que los bordes de la carta resalten del fondo.

### 2. Segmentación (Encontrar las Cartas)
¿Dónde está la carta en la imagen?
- El sistema intenta varios métodos para encontrar contornos: **Thresholding Adaptativo**, **Canny Edge Detection**, y **Separación RGB**.
- **`cv2.findContours`**: Esta función de OpenCV dibuja líneas invisibles buscando cambios bruscos de contraste (por ejemplo, el borde negro de la carta sobre la mesa blanca).
- Filtramos los contornos por **área geométrica**. Si el contorno es diminuto, es "ruido". Si tiene forma de cuadrilátero (`Shapely.Polygon`), es un candidato a carta.

### 3. Transformación de Perspectiva (Enderezar la Carta)
¡Rara vez hacemos una foto perfectamente desde arriba!
- Una vez que sabemos las esquinas del cuadrilátero, aplicamos la **transformación de 4 puntos** (`cv2.warpPerspective`). Esta operación matemática "estira y aplasta" la imagen usando álgebra lineal hasta que la carta queda como un rectángulo perfecto en 2D, tal y como si hubiera sido escaneada.

### 4. Reconocimiento (¿Qué carta es?)
Con la carta ya recortada, necesitamos identificarla rápido.
- En lugar de usar complejas Redes Neuronales Clasificadoras que tardarían horas, extraemos el arte interior de la carta y creamos un **pHash** (Perceptual Hash) usando la librería `imagehash`.
- Calculamos la "Distancia de Hamming" aplicando una operación XOR rápida contra toda nuestra base de datos. ¡La carta de la base de datos con el hash más similar gana!

### 5. Análisis de Metadatos (OCR y Detección de Idioma)
Ya sabemos qué carta es basándonos en el dibujo, ¿pero de qué edición es o en qué idioma está impresa?
- Cortamos regiones específicas de la carta (como la esquina inferior izquierda del borde donde MTG pone el año del Set).
- Pasamos esas regiones minúsculas a **EasyOCR**, que nos revuelve el texto impreso.
- Analizamos la llamada "Type Line" (Línea de Tipo) buscando palabras clave ("Creature", "Criatura", "Créature") para derivar el **idioma** de impresión de la carta.

---

## 🚀 Cómo ejecutar el proyecto en local

### Requisitos previos
Ten instalado Python 3.9+ en tu sistema y asegúrate de clonar este repositorio:

```bash
git clone https://github.com/TuUsuario/magic_card_detectorUPO.git
cd magic_card_detectorUPO
```

### 1. Instalar las dependencias
Instala todas las librerías mencionadas usando el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

*(Nota: Instalar `easyocr` puede requerir una descarga adicional de modelos grandes de IA la primera vez que se ejecute).*

### 2. Ejecutar la Web App
El proyecto incluye un servidor de prueba. Arráncalo con:

```bash
python app.py
```

Ve a tu navegador y entra en `http://127.0.0.1:5001`. Sube una imagen (o varias a la vez para usar el procesamiento Batch) y la interfaz devolverá la imagen corregida con cajas verdes indicando la detección y un CSV si es necesario.

### 3. Ejecutar los Test Locales
Si quieres asegurarte de que la base de datos JSON se carga bien sin arrancar Flask:
```bash
python test.py
```

---

## 📁 Estructura del Proyecto Educativo

```text
.
├── app.py                     # Motor principal de la aplicación web Flask (Rutas e Interfaz)
├── magic_card_detector.py     # 🧠 Aquí está la MAGIA (Algoritmos de CV, OCR y Geometría)
├── README.md                  # Este documento (Tu guía didáctica actual)
├── requirements.txt           # Lista fiel de dependencias necesarias (OpenCV, etc.)
├── test.py                    # Script breve y sencillo de pruebas
├── Script_DB/                 # Base de datos SQLite y JSON de Scryfall de referencia
└── templates/                 # Interfaces HTML/frontend (Página de inicio y Resultados)
```

## 📝 Conclusiones y Práctica Propuesta

Como estudiante de informática o entusiasta de la Inteligencia Artificial, te invito a:
- Revisar la función `four_point_transform` en `magic_card_detector.py`. Juega a alterar el multiplicador y ve qué ocurre con la imagen final.
- Apagar temporalmente el OCR (`OCR_AVAILABLE = False`) en el código y observar la diferencia radical en velocidad. ¡La inferencia OCR es matemáticamente muy intensa comparada con la heurística visual del hashing!
- Intentar añadir un nuevo fallback de segmentación usando "Color Masking" para cartas con bordes blancos de ediciones antiguas.

¡Diviértete explorando el mundo de la Visión por Computador!

---
*License: Este proyecto se distribuye bajo fines de aprendizaje práctico y académico en uso de licenciamiento MIT original adaptado.*
