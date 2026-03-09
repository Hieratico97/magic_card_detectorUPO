"""
Flask Web Application for the Didactic Magic Card Detector.
Serves as the frontend and API gateway for the Computer Vision core.
"""

import os
import io
import base64
import logging
import csv
from collections import Counter
from typing import Tuple, List, Optional, Dict

import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# Importamos nuestro motor de Visión por Computador (El Cerebro)
from magic_card_detector import MagicCardDetector

# --- 1. CONFIGURACIÓN DEL SERVIDOR ---
class Config:
    """Configuración principal para la app Flask."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # Protege contra ataques subiendo archivos gigantes (Max 64 MB)
    REFERENCE_DB_FILE = os.path.join('Script_DB', 'scryfall_db.sqlite')

# --- 2. LOGGING (TRAZABILIDAD) ---
# En un entorno profesional, necesitamos saber qué ocurre sin mirar print()s aislados.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializamos Flask
app = Flask(__name__)
app.config.from_object(Config)

# --- 3. PATRÓN SINGLETON DEL DETECTOR ---
detector: Optional[MagicCardDetector] = None

def get_detector() -> Optional[MagicCardDetector]:
    """
    Instancia el detector de Visión Artificial.
    Usa el patrón 'Singleton' para evitar recargar la Base de Datos SQLite 
    y el detector CLAHE cada vez que un usuario sube una foto.
    """
    global detector
    if detector is not None:
        return detector

    logger.info("Inicializando el Motor MagicCardDetector...")
    if not os.path.exists(app.config['REFERENCE_DB_FILE']):
        logger.critical(f"No se encontró la base de datos en {app.config['REFERENCE_DB_FILE']}")
        return None

    try:
        det_instance = MagicCardDetector()
        det_instance.read_reference_data_from_db(app.config['REFERENCE_DB_FILE'])
        logger.info("Detector inicializado correctamente con indexación Vectorial.")
        detector = det_instance
        return detector
    except Exception as e:
        logger.error(f"Fallo crítico al inicializar el detector: {e}", exc_info=True)
        return None

# Llamamos al detector de inmediato al arrancar el servidor (Warm-up)
get_detector()

# --- 4. FUNCIONES AUXILIARES (HELPERS) ---
def allowed_file(filename: str) -> bool:
    """Valida que la extensión del archivo subido sea segura y esperada."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_single_image(file_storage: FileStorage) -> Tuple[Optional[bytes], Optional[bytes], List[Dict]]:
    """
    Procesa un único archivo de imagen en memoria sin tocar el disco duro.
    
    Args:
        file_storage: Objeto FileStorage de Flask apuntando a la imagen subida.
        
    Returns:
        Tupla con: (bytes_originales, bytes_anotados_procesados, lista_diccionarios_datos)
    """
    det = get_detector()
    if det is None:
        raise RuntimeError("Servicio de Detector de CV no disponible.")

    # 1. REBOBINAR Y LEER EL BUFFER DE MEMORIA
    # Si Flask ya iteró sobre el archivo (por validaciones), el cursor está al final.
    # Necesitamos resetearlo a 0 para poder leer los bytes crudos a numpy.
    file_storage.seek(0)
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    
    if len(file_bytes) == 0: 
        return None, None, []

    # Decodificamos el array de bits puro en una matriz (Imagen BGR) para OpenCV
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_cv is None: 
        return None, None, []

    # Sanitizar por seguridad en caso logueos o guardados posteriores
    filename = secure_filename(file_storage.filename)
    logger.info(f"Procesando imagen: {filename} | Dimensiones: {img_cv.shape}")
    
    # Enviar al motor principal
    return det.process_image_data(img_cv, filename)

# --- 5. RUTAS DE LA INTERFAZ WEB (FRONTEND) ---
@app.route('/')
def index():
    """Ruta principal web. Muestra el formulario subida."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Ruta (POST) llamada al subir el formulario de imágenes."""
    if 'image_file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    files = request.files.getlist('image_file')
    if not files or files[0].filename == '':
        flash('No image selected.', 'error')
        return redirect(url_for('index'))

    results_gallery = []
    total_references = []
    processed_count = 0
    errors = []

    for file in files:
        if file and allowed_file(file.filename):
            try:
                # 1. Pipeline CV Principal
                orig_bytes, ann_bytes, card_data_list = process_single_image(file)
                
                if orig_bytes is None:
                    errors.append(f"{file.filename}: Vacio o corrupto")
                    continue

                # 2. Gestión de Datos: Acumular para CSV posterior
                for card in card_data_list:
                    name = card['name'].strip()
                    set_code = (card['set'] or "??").upper()
                    lang = (card['lang'] or "EN").upper()
                    ref_string = f"{name}-{set_code}-{lang}"
                    total_references.append(ref_string)

                # 3. Preparación Frontend: Ciframos los binarios en texto plano Base64
                # Esto nos permite incrustar la imagen devuelta por OpenCV cruda en el HTML `<img src="data:image...`
                orig_b64 = base64.b64encode(orig_bytes).decode('utf-8')
                res_b64 = base64.b64encode(ann_bytes).decode('utf-8')
                
                results_gallery.append({
                    'original': orig_b64,
                    'processed': res_b64,
                    'filename': file.filename
                })
                processed_count += 1

            except Exception as e:
                logger.error(f"Error procesando {file.filename}: {e}")
                errors.append(str(e))
        else:
            errors.append(f"{file.filename}: Tipo inválido")

    # Final de Bucle de Subida múltiple
    if processed_count == 0:
        flash(f'Ninguna imagen pudo procesarse. Errores: {", ".join(errors[:3])}', 'error')
        return redirect(url_for('index'))

    # CREACIÓN DINÁMICA DE CSV EN MEMORIA (No toca disco)
    try:
        total_counts = Counter(total_references)  # Agrupa "2x Lightning Bolt-LEA-EN"
        # Usar StringIO sirve como "archivo virtual en RAM"
        csv_output = io.StringIO()
        writer = csv.writer(csv_output)
        writer.writerow(['Referencia', 'Cantidad'])
        for ref, count in total_counts.items():
            writer.writerow([ref, count])
        
        csv_string = csv_output.getvalue()
        csv_b64 = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')
    except Exception as e:
        logger.error(f"La compilanción del CSV falló: {e}")
        csv_b64 = None

    return render_template('results.html', 
                           results_gallery=results_gallery,
                           csv_data=csv_b64,
                           filename="batch_scan.csv")

# --- 6. API REST SECUNDARIA ---
@app.route('/api/detect', methods=['POST'])
def api_detect_card():
    """
    Punto de entrada API (Usado normalmente por scripts de postman, bots o móviles).
    A diferencia de la ruta Web que escupe HTML con base64, retorna puro JSON estandarizado.
    """
    if 'file' not in request.files: 
        return jsonify({'error': 'No file provided'}), 400
        
    files = request.files.getlist('file')
    all_cards_data = []

    try:
        for file in files:
            if allowed_file(file.filename):
                _, _, card_data = process_single_image(file)
                all_cards_data.extend(card_data)
        
        return jsonify({
            'status': 'success',
            'count': len(all_cards_data),
            'cards': all_cards_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
