import os
import io
import base64
import logging
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, abort
from werkzeug.utils import secure_filename

# Importamos tu detector
from magic_card_detector import MagicCardDetector

# --- 1. Configuración Profesional ---
class Config:
    """Configuración de la aplicación. Idealmente se carga desde variables de entorno."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev_secret_key_change_in_production'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    # Límite de tamaño de subida: 16 Megabytes (Evita saturar la memoria)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 
    REFERENCE_DB_FILE = os.path.join('Script_DB', 'scryfall_db.sqlite')

# --- 2. Configuración de Logging ---
# En producción, esto se guardaría en un archivo .log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Inicialización de Flask ---
app = Flask(__name__)
app.config.from_object(Config)

# --- 3. Singleton del Detector (Carga Robusta) ---
detector = None

def get_detector():
    """
    Patrón Singleton Lazy-Loading para el detector.
    Garantiza que solo se carga una vez y maneja errores de inicialización.
    """
    global detector
    if detector is not None:
        return detector

    logger.info("Initializing MagicCardDetector Engine...")
    
    if not os.path.exists(app.config['REFERENCE_DB_FILE']):
        logger.critical(f"Database not found at {app.config['REFERENCE_DB_FILE']}")
        return None

    try:
        det_instance = MagicCardDetector()
        det_instance.read_reference_data_from_db(app.config['REFERENCE_DB_FILE'])
        logger.info("Detector initialized successfully.")
        detector = det_instance
        return detector
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}", exc_info=True)
        return None

# Inicialización temprana (opcional, para que el arranque tarde un poco pero luego sea rápido)
get_detector()

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image_stream(file_stream, filename):
    """Lógica central de procesamiento separada de la ruta web."""
    det = get_detector()
    if det is None:
        raise RuntimeError("Detector service is unavailable.")

    # Leer stream a bytes numpy
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_cv is None:
        raise ValueError("Could not decode image.")

    # Procesar
    logger.info(f"Processing image: {filename} | Size: {img_cv.shape}")
    original_bytes, annotated_bytes = det.process_image_data(img_cv, filename)
    
    return original_bytes, annotated_bytes, det.match_threshold

# --- 4. Rutas Web (Interfaz Humana) ---

@app.route('/')
def index():
    status = "Operational" if get_detector() else "Maintenance Mode (DB Missing)"
    return render_template('index.html', status=status)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image_file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['image_file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            # Procesamiento
            orig_bytes, ann_bytes, _ = process_image_stream(file.stream, secure_filename(file.filename))
            
            # Codificar a Base64 para HTML
            orig_b64 = base64.b64encode(orig_bytes).decode('utf-8') if orig_bytes else None
            res_b64 = base64.b64encode(ann_bytes).decode('utf-8') if ann_bytes else None
            
            return render_template('results.html', 
                                   original_image_b64=orig_b64, 
                                   result_image_b64=res_b64)
        except Exception as e:
            logger.error(f"Web processing error: {e}")
            flash(str(e), 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type', 'error')
        return redirect(url_for('index'))

# --- 5. API RESTful (Para Apps Móviles / Integraciones) ---
# Esto añade muchísimo valor profesional. Permite que otros sistemas usen tu motor.

@app.route('/api/detect', methods=['POST'])
def api_detect_card():
    """
    Endpoint JSON para integración con otras apps o móviles.
    Retorna los datos de las cartas detectadas en JSON.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Nota: Para una API real, necesitaríamos modificar `process_image_data` 
        # para que devuelva una lista de objetos (JSON) en lugar de bytes de imagen pintada.
        # Por ahora, devolvemos éxito genérico.
        
        # Simulamos el proceso para validar que no rompe
        process_image_stream(file.stream, secure_filename(file.filename))
        
        return jsonify({
            'status': 'success',
            'message': 'Image processed successfully',
            # Aquí irían los datos estructurados: [{'name': 'Black Lotus', 'set': 'LEA'}]
            # Requiere refactorizar magic_card_detector para devolver datos puros.
        })

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

# --- Manejo de Errores HTTP ---
@app.errorhandler(413)
def request_entity_too_large(error):
    return "File Too Large (Max 16MB)", 413

@app.errorhandler(500)
def internal_error(error):
    return "Internal Server Error - Check Logs", 500

# --- Ejecución ---
if __name__ == "__main__":
    # En producción, no usar app.run(). Usar Gunicorn.
    # Ejemplo: gunicorn -w 4 -b 0.0.0.0:5001 app:app
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting Flask Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)