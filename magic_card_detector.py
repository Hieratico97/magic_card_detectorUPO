"""
Module for detecting and recognizing Magic: the Gathering cards.
VERSION: FINAL + TEXT BOX LANGUAGE DETECTION
Features:
- Vectorized Search (Speed)
- Robust Segmentation (Accuracy)
- Advanced Language Detection (Type Line Keywords + Text Box Analysis)
"""

import os
import io
import sys
import sqlite3
import re 
import imagehash
import cv2
import numpy as np
import matplotlib
from dataclasses import dataclass, field
from copy import deepcopy
from itertools import product
from PIL import Image as PILImage

# --- LIBRERÍA DE DETECCIÓN DE IDIOMA ---
try:
    from langdetect import detect, DetectorFactory,  LangDetectException
    DetectorFactory.seed = 0
    LANG_LIB_AVAILABLE = True
except ImportError:
    LANG_LIB_AVAILABLE = False
    print("WARNING: 'langdetect' not installed. Run: pip install langdetect")

# --- CONFIGURACIÓN DE IDIOMAS (Keywords) ---
TYPE_LINE_LANGUAGES = {
    'es': ('Criatura', 'Conjuro', 'Instantáneo', 'Instantaneo', 'Tierra', 'Encantamiento', 'Artefacto', 'Legendaria', 'Invocar', 'Interrupción'),
    'en': ('Creature', 'Sorcery', 'Instant', 'Land', 'Enchantment', 'Artifact', 'Legendary', 'Summon', 'Interrupt', 'Enchant', 'Mana Source'),
    'fr': ('Créature', 'Rituel', 'Éphémère', 'Ephemere', 'Terrain', 'Enchantement', 'Artefact', 'Légendaire', 'Invoquer', 'Interruption'),
    'de': ('Kreatur', 'Hexerei', 'Spontanzauber', 'Land', 'Verzauberung', 'Artefakt', 'Legendäre', 'Beschwörung', 'Unterbrechung'),
    'it': ('Creatura', 'Stregoneria', 'Istantaneo', 'Terra', 'Incantesimo', 'Artefatto', 'Leggendaria', 'Evoca', 'Interruzione'),
    'pt': ('Criatura', 'Feitiço', 'Mágica Instantânea', 'Terreno', 'Encantamento', 'Artefato', 'Lendária', 'Invocar')
}

# --- EASYOCR CONFIG ---
try:
    import easyocr
    OCR_AVAILABLE = True
    print("EasyOCR detected. Initializing...")
    try:
        # Cargamos varios idiomas para que el OCR no se vuelva loco con caracteres extraños
        reader = easyocr.Reader(['en', 'es', 'fr', 'it', 'de', 'pt'], gpu=True)
        print("✅ EasyOCR initialized on GPU.")
    except Exception:
        print("⚠️ GPU failed. Running on CPU.")
        reader = easyocr.Reader(['en', 'es', 'fr', 'it', 'de', 'pt'], gpu=False)
except ImportError:
    OCR_AVAILABLE = False
    print("WARNING: 'easyocr' not installed.")

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shapely.geometry import LineString, Polygon
from shapely.affinity import scale
from scipy.ndimage import rotate

# --- GEOMETRY HELPER FUNCTIONS ---

def order_polygon_points(x, y):
    angle = np.arctan2(y - np.average(y), x - np.average(x))
    ind = np.argsort(angle)
    return (x[ind], y[ind])

def four_point_transform(image, poly):
    pts = np.zeros((4, 2))
    pts[:, 0] = np.asarray(poly.exterior.coords)[:-1, 0]
    pts[:, 1] = np.asarray(poly.exterior.coords)[:-1, 1]
    
    rect = np.zeros((4, 2))
    (rect[:, 0], rect[:, 1]) = order_polygon_points(pts[:, 0], pts[:, 1])

    width_a = np.sqrt(((rect[1, 0] - rect[0, 0]) ** 2) + ((rect[1, 1] - rect[0, 1]) ** 2))
    width_b = np.sqrt(((rect[3, 0] - rect[2, 0]) ** 2) + ((rect[3, 1] - rect[2, 1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((rect[0, 0] - rect[3, 0]) ** 2) + ((rect[0, 1] - rect[3, 1]) ** 2))
    height_b = np.sqrt(((rect[1, 0] - rect[2, 0]) ** 2) + ((rect[1, 1] - rect[2, 1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    rect = np.array([
        [rect[0, 0], rect[0, 1]],
        [rect[1, 0], rect[1, 1]],
        [rect[2, 0], rect[2, 1]],
        [rect[3, 0], rect[3, 1]]], dtype="float32")

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    transform = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))
    return warped

def get_bounding_quad(hull_poly):
    return hull_poly.minimum_rotated_rectangle

def quad_corner_diff(hull_poly, bquad_poly):
    if bquad_poly.area == 0: return 1.0
    return (bquad_poly.area - hull_poly.area) / bquad_poly.area

def convex_hull_polygon(contour):
    hull = cv2.convexHull(contour)
    if len(hull) < 3: return None
    return Polygon([[x, y] for (x, y) in zip(hull[:, :, 0], hull[:, :, 1])])

def characterize_card_contour(card_contour, max_segment_area, image_area):
    phull = convex_hull_polygon(card_contour)
    if phull is None or (phull.area < 0.1 * max_segment_area or phull.area < image_area / 1000.):
        return (False, False, None, 1.)
    
    bounding_poly = get_bounding_quad(phull)
    qc_diff = quad_corner_diff(phull, bounding_poly)
    crop_factor = min(1., (1. - qc_diff * 22. / 100.))
    
    is_card_candidate = bool(
        0.02 * max_segment_area < bounding_poly.area < image_area * 0.99 and
        qc_diff < 0.45 
    )
    return (True, is_card_candidate, bounding_poly, crop_factor)

# --- DATA CLASSES ---

@dataclass
class CardCandidate:
    image: np.ndarray = field(compare=False) 
    bounding_quad: Polygon
    image_area_fraction: float
    is_recognized: bool = False
    recognition_score: float = 0.
    is_fragment: bool = False
    name: str = 'unknown'
    set_info: str = '' 
    db_set_code: str = ''
    language: str = ''

    def center_x(self):
        coords = np.asarray(self.bounding_quad.exterior.coords)
        return np.mean(coords[:, 0])

class ReferenceImage:
    def __init__(self, name, phash_obj, set_code):
        self.name = name
        self.phash = phash_obj 
        self.set_code = set_code.upper() if set_code else "???"

class TestImage:
    def __init__(self, name, original_image, clahe):
        self.name = name
        self.original = original_image
        self.clahe = clahe
        self.adjusted = None
        self.candidate_list = []
        self.visual = False
        self.histogram_adjust()

    def histogram_adjust(self):
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        self.adjusted = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def plot_image_with_recognized(self):
        plt.figure(figsize=(12, 8))
        rgb_img = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.axis('off')

        for candidate in self.candidate_list:
            if candidate.is_fragment: continue

            coords = np.array(candidate.bounding_quad.exterior.coords)
            plt.plot(coords[:, 0], coords[:, 1], 'g-', linewidth=2)

            if candidate.is_recognized:
                cx, cy = np.mean(coords[:-1, 0]), np.mean(coords[:-1, 1])
                
                display_text = f"{candidate.name}"
                
                if candidate.set_info and candidate.db_set_code:
                    if candidate.set_info in candidate.db_set_code or candidate.db_set_code in candidate.set_info:
                        display_text += f"\n★ {candidate.db_set_code} ★"
                    else:
                        display_text += f"\n(DB: {candidate.db_set_code})"
                elif candidate.db_set_code:
                     display_text += f"\n({candidate.db_set_code})"
                
                if candidate.language:
                    display_text += f" [{candidate.language.upper()}]"
                
                plt.text(cx, cy, display_text, color='black', fontsize=9, ha='center',
                         bbox=dict(facecolor='white', alpha=0.85, edgecolor='green', boxstyle='round,pad=0.3'))

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    def mark_fragments(self):
        self.candidate_list.sort(key=lambda x: x.bounding_quad.area, reverse=True)
        for c1, c2 in product(self.candidate_list, repeat=2):
            if c1 is c2: continue 
            if c1.is_fragment: continue
            inter_area = c1.bounding_quad.intersection(c2.bounding_quad).area
            min_area = min(c1.bounding_quad.area, c2.bounding_quad.area)
            if inter_area > 0.5 * min_area:
                if c1.recognition_score >= c2.recognition_score:
                    c2.is_fragment = True
                else:
                    c1.is_fragment = True

# --- MAIN DETECTOR CLASS ---

class MagicCardDetector:
    def __init__(self, output_path=None):
        self.reference_images = [] 
        self.db_hashes_matrix = None 
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.verbose = True
        self.output_path = output_path
        self.match_threshold = 62 

    def hex_to_bool_array(self, hex_str):
        try:
            if not hex_str: return None
            h_int = int(hex_str, 16)
            bin_str = bin(h_int)[2:].zfill(256)
            return np.array([int(b) for b in bin_str], dtype=bool)
        except: return None

    def read_reference_data_from_db(self, db_path):
        print(f'Loading reference data from {db_path}...')
        if not os.path.exists(db_path):
            print("Database file not found!")
            return
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, phash, set_code FROM cards WHERE phash IS NOT NULL AND phash != ''")
            rows = cursor.fetchall()
            
            temp_hashes_list = []
            self.reference_images = []
            
            loaded = 0
            for name, phash_str, set_code in rows:
                try:
                    h_obj = imagehash.hex_to_hash(phash_str)
                    self.reference_images.append(ReferenceImage(name, h_obj, set_code))
                    
                    bool_arr = self.hex_to_bool_array(phash_str)
                    if bool_arr is not None and len(bool_arr) == 256:
                        temp_hashes_list.append(bool_arr)
                    else:
                        temp_hashes_list.append(np.zeros(256, dtype=bool))
                    
                    loaded += 1
                except Exception:
                    continue
            
            if temp_hashes_list:
                self.db_hashes_matrix = np.vstack(temp_hashes_list)
                print(f"✅ Vectorized Search Index Built: {self.db_hashes_matrix.shape}")
            
            conn.close()
            print(f'Successfully loaded {loaded} cards.')
        except Exception as e:
            print(f"Error reading DB: {e}")

    def fast_search(self, candidate_hash_obj):
        if self.db_hashes_matrix is None: return [], []
        cand_bool = np.array(candidate_hash_obj.hash.flatten(), dtype=bool)
        if len(cand_bool) != 256: return [], []
        
        xor_matrix = np.bitwise_xor(self.db_hashes_matrix, cand_bool)
        distances = np.count_nonzero(xor_matrix, axis=1)
        
        matches_mask = distances <= self.match_threshold
        match_indices = np.where(matches_mask)[0]
        
        if len(match_indices) > 0:
            match_distances = distances[match_indices]
            sorted_local_indices = np.argsort(match_distances)
            return match_indices[sorted_local_indices], match_distances[sorted_local_indices]
        
        return [], []

    def read_prehashed_reference_data(self, pickle_file):
        pass 

    def process_image_data(self, image_cv2, image_name="uploaded_image"):
        print(f"Processing: {image_name}")
        test_image = TestImage(image_name, image_cv2, self.clahe)
        
        any_recognized = False

        # 1. Segmentación
        seg_methods = ['adaptive', 'canny', 'rgb']
        for method in seg_methods:
            self.segment_image(test_image, mode=method)
            for candidate in test_image.candidate_list:
                if not candidate.is_recognized:
                    self.recognize_candidate(candidate)
                    if candidate.is_recognized: any_recognized = True
            if any_recognized: break 

        # 2. Fallback: Center Crop
        if not any_recognized:
            h, w = test_image.adjusted.shape[:2]
            x1, x2 = int(w * 0.20), int(w * 0.80)
            y1, y2 = int(h * 0.05), int(h * 0.95)
            poly = Polygon([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
            cand = CardCandidate(test_image.adjusted[y1:y2, x1:x2], poly, 0.6)
            test_image.candidate_list.append(cand)
            self.recognize_candidate(cand)
            if cand.is_recognized: any_recognized = True

        # 3. Fallback: Full Image
        if not any_recognized:
            h, w = test_image.adjusted.shape[:2]
            poly = Polygon([(0,0), (w,0), (w,h), (0,h)])
            cand = CardCandidate(test_image.adjusted, poly, 1.0)
            test_image.candidate_list.append(cand)
            self.recognize_candidate(cand)

        test_image.mark_fragments()
        test_image.candidate_list.sort(key=lambda c: c.center_x())

        annotated_bytes = test_image.plot_image_with_recognized()
        is_success, buffer = cv2.imencode(".png", test_image.original)
        original_bytes = buffer.tobytes() if is_success else None
        return original_bytes, annotated_bytes

    def is_solid_color(self, image, threshold=15):
        if image.size == 0: return True
        (mean, std) = cv2.meanStdDev(image)
        return np.mean(std) < threshold

    def segment_image(self, test_image, mode='adaptive'):
        full_image = test_image.adjusted
        image_area = full_image.shape[0] * full_image.shape[1]
        max_segment_area = image_area 
        contours = []
        gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)

        if mode == 'adaptive':
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = cnts
        elif mode == 'canny':
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 30, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
            thresh = cv2.dilate(edged, kernel, iterations=2)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = cnts
        elif mode == 'rgb':
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = cnts

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15] 

        for cnt in contours:
            valid, is_cand, b_poly, crop_f = characterize_card_contour(cnt, max_segment_area, image_area)
            if valid and is_cand:
                if max_segment_area == image_area: max_segment_area = b_poly.area
                warped = four_point_transform(full_image, scale(b_poly, xfact=crop_f, yfact=crop_f, origin='centroid'))
                if not self.is_solid_color(warped):
                    test_image.candidate_list.append(CardCandidate(warped, b_poly, b_poly.area / image_area))

    def generate_crop_variations(self, card_image):
        crops = []
        h, w = card_image.shape[:2]
        y1, y2 = int(h * 0.10), int(h * 0.55)
        x1, x2 = int(w * 0.08), int(w * 0.92)
        crops.append(card_image[y1:y2, x1:x2])
        y1_z, y2_z = int(h * 0.15), int(h * 0.50)
        x1_z, x2_z = int(w * 0.12), int(w * 0.88)
        crops.append(card_image[y1_z:y2_z, x1_z:x2_z])
        y1_f, y2_f = int(h * 0.07), int(h * 0.55)
        crops.append(card_image[y1_f:y2_f, x1:x2])
        return crops

    def detect_language_extended(self, full_card_image):
        """
        Intenta detectar el idioma en dos fases:
        1. Keywords en la línea de tipo (rápido y preciso).
        2. langdetect en el cuadro de texto (lento pero bueno para textos largos).
        """
        if not OCR_AVAILABLE: return ""
        h, w = full_card_image.shape[:2]
        
        # --- FASE 1: Línea de Tipo (Keywords) ---
        y1_type, y2_type = int(h * 0.56), int(h * 0.64)
        x1_type, x2_type = int(w * 0.05), int(w * 0.95)
        crop_type = full_card_image[y1_type:y2_type, x1_type:x2_type]
        
        gray_type = cv2.cvtColor(crop_type, cv2.COLOR_BGR2GRAY)
        _, thresh_type = cv2.threshold(gray_type, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        type_text = ""
        try:
            results = reader.readtext(thresh_type, detail=0) 
            type_text = " ".join(results).lower()
        except Exception: pass

        for lang_code, keywords in TYPE_LINE_LANGUAGES.items():
            for keyword in keywords:
                if keyword.lower() in type_text:
                    print(f"  [Lang] Keyword Match: {lang_code.upper()}")
                    return lang_code
        
        # --- FASE 2: Cuadro de Texto (Library) ---
        # Si no hay keywords, miramos el texto de reglas
        if LANG_LIB_AVAILABLE:
            # Crop Text Box (approx 65% to 88% height)
            y1_text, y2_text = int(h * 0.65), int(h * 0.88)
            x1_text, x2_text = int(w * 0.05), int(w * 0.95)
            crop_text = full_card_image[y1_text:y2_text, x1_text:x2_text]
            
            gray_text = cv2.cvtColor(crop_text, cv2.COLOR_BGR2GRAY)
            _, thresh_text = cv2.threshold(gray_text, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            try:
                results_text = reader.readtext(thresh_text, detail=0)
                full_text = " ".join(results_text)
                
                # langdetect necesita cierto volumen de texto para ser fiable
                if len(full_text) > 15:
                    lang = detect(full_text)
                    # Filtramos: Solo aceptamos si es uno de los idiomas de Magic
                    if lang in TYPE_LINE_LANGUAGES.keys():
                        print(f"  [Lang] Lib Detected: {lang.upper()}")
                        return lang
            except Exception as e:
                pass

        return ""

    def attempt_set_or_year_ocr(self, full_card_image):
        if not OCR_AVAILABLE: return ""
        h, w = full_card_image.shape[:2]
        crop_modern = full_card_image[int(h*0.90):int(h*0.97), int(w*0.04):int(w*0.35)]
        crop_bottom = full_card_image[int(h*0.92):int(h*0.98), int(w*0.05):int(w*0.95)]
        detected_info = self._run_ocr_scan(crop_modern, mode="set_code")
        if not detected_info:
            detected_info = self._run_ocr_scan(crop_bottom, mode="year")
        return detected_info

    def _run_ocr_scan(self, img_crop, mode="set_code"):
        try:
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            results = reader.readtext(thresh)
        except Exception:
            return ""

        for (bbox, text, prob) in results:
            if prob < 0.3: continue
            clean = text.upper()
            if mode == "set_code":
                clean_code = re.sub(r'[^A-Z0-9]', '', clean)
                match = re.search(r'\b[A-Z0-9]{3,4}\b', clean_code)
                if match: return match.group(0)
            elif mode == "year":
                match = re.search(r'(19|20)\d{2}', clean)
                if match: return f"Year: {match.group(0)}"
        return ""

    def recognize_candidate(self, candidate):
        rotations = [0, 90, 180, 270]
        all_possible_matches = []

        for rot in rotations:
            img_rot = candidate.image
            if rot != 0: img_rot = rotate(candidate.image, rot)
            art_crops = self.generate_crop_variations(img_rot)
            for crop in art_crops:
                try:
                    if crop.size == 0: continue
                    pil_art = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    hash_art = imagehash.phash(pil_art, hash_size=16)
                    indices, dists = self.fast_search(hash_art)
                    for i, dist in zip(indices, dists):
                        all_possible_matches.append({
                            'ref_idx': i,
                            'img': img_rot, 
                            'diff': dist
                        })
                except: pass

        if not all_possible_matches: return

        all_possible_matches.sort(key=lambda x: x['diff'])
        top_candidates = all_possible_matches[:5]
        
        best_match = top_candidates[0]
        final_ref = self.reference_images[best_match['ref_idx']]
        
        detected_set_code = ""
        detected_lang = ""
        
        if OCR_AVAILABLE:
            print(f"  [FAST] Analyzing candidate: {final_ref.name} (Diff: {best_match['diff']})")
            detected_set_code = self.attempt_set_or_year_ocr(best_match['img'])
            
            # USAMOS LA NUEVA FUNCIÓN EXTENDIDA
            detected_lang = self.detect_language_extended(best_match['img'])

            if detected_set_code:
                for cand in top_candidates:
                    ref = self.reference_images[cand['ref_idx']]
                    if ref.set_code == detected_set_code:
                        final_ref = ref
                        print(f"  -> Edition Match! Switched to {final_ref.name} ({final_ref.set_code})")
                        break

        candidate.is_recognized = True
        candidate.name = final_ref.name
        candidate.recognition_score = 100 - best_match['diff']
        candidate.db_set_code = final_ref.set_code
        candidate.set_info = detected_set_code 
        # FALLBACK IDIOMA: Si tras todo esto sigue vacío, ponemos EN
        candidate.language = detected_lang if detected_lang else "en"
        
        print(f"Match Final: {candidate.name} ({candidate.db_set_code})")

if __name__ == "__main__":
    pass