import os
import io  # NECESARIO PARA CSV
import base64
import logging
import csv
from collections import Counter
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# Importamos tu detector
from magic_card_detector import MagicCardDetector

# --- 1. Configuración ---
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev_secret_key_change_in_production'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB
    REFERENCE_DB_FILE = os.path.join('Script_DB', 'scryfall_db.sqlite')

# --- 2. Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Inicialización ---
app = Flask(__name__)
app.config.from_object(Config)

# --- 3. Detector Singleton ---
detector = None

def get_detector():
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

get_detector()

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_single_image(file_storage):
    det = get_detector()
    if det is None: raise RuntimeError("Detector unavailable.")

    # 1. REBOBINAR ARCHIVO (Corrección Clave)
    file_storage.seek(0)
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    
    if len(file_bytes) == 0: return None, None, []

    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_cv is None: return None, None, []

    filename = secure_filename(file_storage.filename)
    logger.info(f"Processing image: {filename} | Size: {img_cv.shape}")
    
    return det.process_image_data(img_cv, filename)

# --- Rutas ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
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
                orig_bytes, ann_bytes, card_data_list = process_single_image(file)
                
                if orig_bytes is None:
                    errors.append(f"{file.filename}: Empty/Corrupt")
                    continue

                # Acumular para CSV
                for card in card_data_list:
                    name = card['name'].strip()
                    set_code = (card['set'] or "??").upper()
                    lang = (card['lang'] or "EN").upper()
                    ref_string = f"{name}-{set_code}-{lang}"
                    total_references.append(ref_string)

                # Guardar Galería
                orig_b64 = base64.b64encode(orig_bytes).decode('utf-8')
                res_b64 = base64.b64encode(ann_bytes).decode('utf-8')
                
                results_gallery.append({
                    'original': orig_b64,
                    'processed': res_b64,
                    'filename': file.filename
                })
                processed_count += 1

            except Exception as e:
                logger.error(f"Processing error {file.filename}: {e}")
                errors.append(str(e))
        else:
            errors.append(f"{file.filename}: Invalid Type")

    if processed_count == 0:
        flash(f'No images processed. Errors: {", ".join(errors[:3])}', 'error')
        return redirect(url_for('index'))

    # GENERAR CSV
    try:
        total_counts = Counter(total_references)
        csv_output = io.StringIO()
        writer = csv.writer(csv_output)
        writer.writerow(['Referencia', 'Cantidad'])
        for ref, count in total_counts.items():
            writer.writerow([ref, count])
        
        csv_string = csv_output.getvalue()
        csv_b64 = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')
    except Exception as e:
        logger.error(f"CSV generation failed: {e}")
        csv_b64 = None

    return render_template('results.html', 
                           results_gallery=results_gallery,
                           csv_data=csv_b64,
                           filename="batch_scan.csv")

@app.route('/api/detect', methods=['POST'])
def api_detect_card():
    if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
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