import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for, flash

from magic_card_detector import MagicCardDetector

# --- Configuration ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
REFERENCE_DB_FILE = 'scryfall_db.sqlite'  # Cambiado para usar SQLite

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = 'your secret key here' # TBD

# --- Load MagicCardDetector ONCE ---
print("Initializing Magic Card Detector...")
detector = MagicCardDetector()

def load_reference_data():
    """
    Intenta cargar los datos de referencia desde SQLite, 
    si no existe, usa el archivo pickle como fallback
    """
    # Primero intenta con la base de datos SQLite
    if os.path.exists(REFERENCE_DB_FILE):
        try:
            detector.read_reference_data_from_db(REFERENCE_DB_FILE)
            print(f"Successfully loaded reference data from {REFERENCE_DB_FILE}")
            return True
        except Exception as e:
            print(f"Error loading from SQLite database: {e}")
            print("Falling back to pickle file...")
    
    # Fallback al archivo pickle
    pickle_file = 'alpha_reference_phash.dat'
    if os.path.exists(pickle_file):
        try:
            detector.read_prehashed_reference_data(pickle_file)
            print(f"Successfully loaded reference data from {pickle_file}")
            return True
        except Exception as e:
            print(f"Error loading from pickle file: {e}")
    
    print("ERROR: No reference data found!")
    return False

# Cargar datos de referencia
if not load_reference_data():
    detector = None  # Deshabilitar detector si no hay datos de referencia

# --- Helper Function ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/')
def index():
    """Serves the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handles image upload, processing, and displays results."""
    if detector is None:
         flash('Card Detector could not be initialized (Reference data missing?). Cannot process image.', 'error')
         return redirect(url_for('index'))

    if 'image_file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('index'))

    file = request.files['image_file']

    if file.filename == '':
        flash('No image selected for uploading.', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            # Read image file stream into OpenCV
            filestr = file.read()
            npimg = np.frombuffer(filestr, np.uint8)
            img_cv = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if img_cv is None:
                 flash('Could not decode image. Please upload a valid image file (JPG, PNG).', 'error')
                 return redirect(url_for('index'))

            print(f"Image '{file.filename}' loaded successfully. Processing...")

            # Process the image using the detector instance
            # This now returns original_bytes, annotated_bytes
            original_bytes, annotated_bytes = detector.process_image_data(img_cv, file.filename)

            print("Processing complete. Encoding images for display...")

            # Encode images to Base64 for embedding in HTML
            original_b64 = base64.b64encode(original_bytes).decode('utf-8') if original_bytes else None
            result_b64 = base64.b64encode(annotated_bytes).decode('utf-8') if annotated_bytes else None

            # Render the results page
            return render_template('results.html',
                                   original_image_b64=original_b64,
                                   result_image_b64=result_b64)

        except Exception as e:
            print(f"An error occurred during processing: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback to server console
            flash(f'An error occurred during processing: {e}', 'error')
            return redirect(url_for('index'))

    else:
        flash('Allowed image types are -> png, jpg, jpeg', 'error')
        return redirect(url_for('index'))

# --- Run the App ---
if __name__ == "__main__":
    # Use debug=True only for development
    # For production, use a proper WSGI server like Gunicorn or Waitress
    # Example: gunicorn -w 4 app:app
    app.run(debug=True, host='0.0.0.0', port=5001) # Makes it accessible on your network