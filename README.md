# 🎓 Magic Card Detector UPO - Proyecto Guiado de Visión por Computador

¡Bienvenido/a a este proyecto! Este repositorio no solo es una herramienta funcional para escanear y reconocer cartas de Magic: The Gathering (MTG), sino también un **proyecto de referencia didáctica** diseñado para aprender **Visión por Computador**.

Si alguna vez te has preguntado cómo tu teléfono o un escáner pueden detectar un documento, recortarlo en perspectiva, analizar si hay texto y qué dice, o saber si la imagen corresponde a un objeto en su base de datos... **este proyecto te enseña todo ese proceso paso a paso**.

---

## 🌟 Novedades de este Fork (vs Original)

Este proyecto nace como un fork de [tmikonen/magic_card_detector](https://github.com/tmikonen/magic_card_detector), pero ha sido **profundamente reescrito y mejorado** para ser más rápido, preciso y para su uso productivo y educativo. Las principales mejoras implementadas son:

1. **Búsqueda Vectorial Ultrarrápida**: El original iteraba carta por carta para calcular similitudes de hashes de forma secuencial. Aquí hemos vectorizado la base de datos completa de Hashes en matrices booleanas (NumPy), lo que nos permite calcular la *Distancia de Hamming* masivamente ejecutando operaciones matemáticas XOR de forma simultánea e instantánea.
2. **Base de Datos Robusta (SQLite)**: Se ha abandonado el sistema de archivos simples. Ahora el proyecto interactúa directamente con una base de datos local generada a partir de los datos oficiales de *Scryfall*, haciendo las consultas más seguras y estructuradas.
3. **Módulo OCR de Idioma y Edición**: Integración nativa con Inteligencia Artificial vía **EasyOCR**. El sistema es capaz de "leer" el set de impresión de la esquina de la carta y deducir el idioma analizando la tipografía de la tarjeta (Ej. *Creature* vs *Criatura*).
4. **Múltiples Estrategias de Segmentación**: El repositorio original dependía de contornos simples, fallando sobre fondos con mucho ruido. Este fork implementa una cascada de algoritmos: primero intenta Threshold Adaptativo, luego Detección Canny, y luego separación de escalas de colores hasta encontrar la carta.
5. **Aplicación Web "Batch" y Exportación CSV**: La aplicación Flask embebida ahora maneja las imágenes completamente en memoria (Base64) en lugar de depender del disco duro. Soporta la subida de un ramillete grande de fotos a la vez y agrega todos los resultados en un archivo CSV descargable, automatizando tareas de inventario.

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
- El sistema implementa una **Cascada de Segmentación** (Adaptive -> Canny -> RGB) prioritaria porque ninguna iluminación es perfecta.
- **Thresholding Adaptativo GAUSSIAN_C**: A diferencia de un binarizado global (donde todo píxel $> 127$ es blanco), el Gaussiano calcula el umbral de cada píxel basándose en un bloque vecino (por ejemplo de $11 \times 11$). Esto soporta sombras gradadas y anillos de luz de lámparas sobre la mesa.
- **Canny Edge Detection**: Si lo adaptativo falla, se usa la derivada direccional de OpenCV (`cv2.Canny`) para identificar cambios estructurales bruscos (bordes físicos marcados). Se dilatan estos bordes usando operaciones morfológicas (`cv2.dilate`) para agrupar huecos (cerrar el contorno de la carta rota o desgastada).
- **Aproximación Poligonal**: Usamos `cv2.findContours` para extraer la malla. Finalmente, `Shapely` aplica su test de Área y Convex Hull para verificar matemáticamente si representa un cuadrilátero válido (y descartar monedas, dados o contadores presentes en la mesa).

### 3. Transformación de Perspectiva (Enderezar la Carta)
¡Rara vez hacemos una foto perfectamente desde arriba ("Top-down")!
- El módulo aplica un algoritmo clásico de OpenCV llamado **Four Point Transform**.
- **Cálculo Euclidiano**: Calculamos la longitud de cada arista del cuadrilátero original (Lado A, Lado B, Base, Altura) por Pitágoras $\sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$. Con esto determinamos la "resolución máxima proyectada" (El ancho de la carta que enderezaremos será la longitud más larga del propio polígono de la foto).
- **Proyección Plana (`cv2.warpPerspective`)**: Al proporcionar las coordenadas de las 4 esquinas deformes de la foto y nuestras 4 referencias finales de un rectángulo canónico $(0,0)$, la función calcula una **Matriz de Transformación Afín $3\times3$**. Deforma píxel a píxel la imagen base hasta convertir nuestro ángulo de teléfono oblicuo en una "cámara cenital perfecta".

### 4. Reconocimiento Vectorial (¿Qué carta es?)
A la carta ya recortada, aplicamos **Perceptual Hashing (pHash)** para su identificación visual frente a fallos.
- **Hashing vs Criptografía**: Al contrario que MD5 o SHA-156, donde cambiar 1 píxel rompe el hash resultante, *pHash* escala la imagen (a $32\times32$), la convierte a grises y aplica una **Transformada de Coseno Discreta (DCT)** para guardar solo frecuencias bajas estructurales (el "esbozo" básico).
- El cálculo resulta en una huella digital (*hash*) de 64 o 256-bits. Cambios de brillo, recortes minúsculos o desenfoques ligeros mantendrán casi inalterado el array de bits.
- **Vectorización NumPy de Alta Velocidad**: Nuestra Base de datos SQLite contiene $~12.000+$ hashes de Scryfall. En el código cargamos todos usando `np.vstack()` hasta formar una inmensa matriz Booleana ($N \times 256$). Para comparar la carta introducida por el usuario, hacemos `np.bitwise_xor` (XOR lógico simultáneo) sobre ella. La operación *np.count_nonzero* sobre la matriz nos dirá de una tajada **la Distancia de Hamming** de todas las miles de cartas. Todo ocurre en ms en procesador multinúcleo.

### 5. Análisis de Metadatos (OCR y Detección de Idioma)
Sabiendo el "Brazalete de dibujo" de la carta, sacamos información oculta (edición y el idioma).
- En caso de varias cartas de Magic con el mismo arte interior pero distinta edición, utilizamos **Inteligencia Artificial (EasyOCR)** sobre regiones minúsculas extrañamente ubicadas (ej.: la esquina 5% inferior izquierda: El año o set ID).
- Extraemos la región "Type Line" (*Criatura, Stregoneria, Artefatto*) y le extraemos texto, para pasarlo como entrada directa heurística al modulo idiomático nativo (En Python: `langdetect` + `Regex`) lo que nos revelará finalmente la impresión real que tiene el alumno entre las manos.

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
