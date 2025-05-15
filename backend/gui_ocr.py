from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO
import easyocr
import re
import pytesseract
from google.cloud import firestore
import uuid
import os
import base64
import json
import tempfile
from pathlib import Path
import time
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Set
import traceback
from dataclasses import dataclass, asdict, field
import math
from collections import defaultdict, deque

# Environment variable configuration
MODEL_PATH = os.environ.get('MODEL_PATH', './models')
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')
DATA_DIR = os.environ.get('DATA_PATH', './data')
TEMP_DIR = os.environ.get('TEMP_DIR', './temp')
MAX_FRAMES_TO_PROCESS = int(os.environ.get('MAX_FRAMES', '300'))  # Maximum frames to process from video
FRAME_SAMPLE_RATE = int(os.environ.get('FRAME_SAMPLE_RATE', '5'))  # Process every Nth frame
MIN_DETECTION_CONFIDENCE = float(os.environ.get('MIN_DETECTION_CONFIDENCE', '0.35'))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
PATTERNS_FILE = os.path.join(DATA_DIR, 'license_patterns.json')
CORRECTIONS_FILE = os.path.join(DATA_DIR, 'character_corrections.json')

# Initialize FastAPI app
app = FastAPI(
    title="PlateVision API",
    description="API for license plate detection and recognition from images and videos",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firestore
try:
    db = firestore.Client()
    firestore_enabled = True
except Exception as e:
    print(f"Firestore initialization failed: {e}")
    firestore_enabled = False

# Load YOLO models
try:
    vehicle_detector = YOLO(f'{MODEL_PATH}/yolo11m.pt')
    license_plate_detector = YOLO(f'{MODEL_PATH}/license_plate_detector.pt')
    models_loaded = True
except Exception as e:
    print(f"Model loading failed: {e}")
    models_loaded = False

# Initialize OCR engines
ocr_engines = {}
try:
    # Primary OCR engine
    ocr_engines['primary'] = easyocr.Reader(['en'], gpu=True)
    
    # Check if Tesseract is available
    pytesseract.get_tesseract_version()
    tesseract_available = True
except Exception as e:
    print(f"OCR initialization warning: {e}")
    tesseract_available = False
    if 'primary' not in ocr_engines:
        print("Critical error: No OCR engines available")
        models_loaded = False

# Load or initialize character corrections
try:
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, 'r') as f:
            CHARACTER_CORRECTIONS = json.load(f)
    else:
        # Initialize with basic common OCR errors
        CHARACTER_CORRECTIONS = {
            '@': ['D', 'O'],
            '0': ['O', 'D'],
            'O': ['0', 'D'],
            'I': ['1', 'L'],
            '1': ['I', 'L'],
            'S': ['5', '8'],
            '5': ['S'],
            'Z': ['2', '7'],
            '2': ['Z'],
            'B': ['8', 'R'],
            '8': ['B'],
            'G': ['6', 'C'],
            '6': ['G'],
            'Q': ['O', '0'],
            'U': ['V', 'W'],
            'V': ['U', 'W'],
            'M': ['N', 'W'],
            'N': ['M'],
            # Noise characters that might be mistakenly detected
            ' ': [''],
            '-': [''],
            '.': [''],
            ':': [''],
            ';': [''],
            ',': [''],
            '*': [''],
            '/': [''],
            '\\': [''],
            '|': [''],
            '"': [''],
            "'": [''],
            '`': [''],
            '=': [''],
            '+': [''],
            '!': ['1', 'I']
        }
        # Save initial corrections
        with open(CORRECTIONS_FILE, 'w') as f:
            json.dump(CHARACTER_CORRECTIONS, f, indent=2)
except Exception as e:
    print(f"Character corrections initialization error: {e}")
    # Fallback to basic corrections
    CHARACTER_CORRECTIONS = {
        '@': ['D', 'O'],
        '0': ['O', 'D'],
        'I': ['1', 'L'],
        'S': ['5', '8'],
    }

# Load or initialize license plate patterns
try:
    if os.path.exists(PATTERNS_FILE):
        with open(PATTERNS_FILE, 'r') as f:
            LICENSE_PATTERNS = json.load(f)
    else:
        # Initialize with some common patterns
        LICENSE_PATTERNS = {
            'generic': {
                'patterns': [
                    r'^[A-Z0-9]{5,9}$',                    # Generic alphanumeric
                    r'^[A-Z]{1,3}[0-9]{3,5}$',             # Letters followed by numbers
                    r'^[0-9]{1,3}[A-Z]{1,3}[0-9]{1,4}$',   # Numbers-Letters-Numbers
                    r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,3}$'    # Letters-Numbers-Letters
                ],
                'examples': [],
                'confidence': 0.6
            },
            'country_specific': {
                'US': {
                    'patterns': [
                        r'^[A-Z0-9]{5,8}$'                 # US plates
                    ],
                    'examples': [],
                    'confidence': 0.7
                },
                'IN': {
                    'patterns': [
                        r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$', # Indian format
                        r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3}$'  # Shorter Indian format
                    ],
                    'state_prefixes': [
                        'AP', 'AR', 'AS', 'BR', 'CG', 'DL', 'GA', 'GJ', 'HR', 'HP', 
                        'JK', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 
                        'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB'
                    ],
                    'examples': [],
                    'confidence': 0.75
                },
                'UK': {
                    'patterns': [
                        r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$'      # UK format
                    ],
                    'examples': [],
                    'confidence': 0.75
                }
            },
            'confidence_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
            }
        }
        # Save initial patterns
        with open(PATTERNS_FILE, 'w') as f:
            json.dump(LICENSE_PATTERNS, f, indent=2)
except Exception as e:
    print(f"License patterns initialization error: {e}")
    # Fallback to simple patterns
    LICENSE_PATTERNS = {
        'generic': {
            'patterns': [r'^[A-Z0-9]{5,9}$'],
            'confidence': 0.5
        }
    }

@dataclass
class OCRResult:
    """Store OCR result with metadata"""
    text: str
    confidence: float
    method: str
    preprocessing: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class PlateDetection:
    """Store information about a license plate detection"""
    plate_text: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    frame_number: int
    timestamp: float
    vehicle_bbox: Optional[List[int]] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class PlateTracker:
    """Track a unique license plate across video frames"""
    plate_id: str
    plate_text: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: float
    last_seen_time: float
    detections: List[PlateDetection] = field(default_factory=list)
    confidence: float = 0.0
    best_detection: Optional[PlateDetection] = None
    count: int = 0
    
    def update(self, detection: PlateDetection):
        """Update tracker with a new detection"""
        self.detections.append(detection)
        self.last_seen_frame = max(self.last_seen_frame, detection.frame_number)
        self.last_seen_time = max(self.last_seen_time, detection.timestamp)
        self.count += 1
        
        # Update best detection if this one has higher confidence
        if self.best_detection is None or detection.confidence > self.best_detection.confidence:
            self.best_detection = detection
            self.confidence = detection.confidence
    
    def to_dict(self):
        return {
            "plate_id": self.plate_id,
            "plate_text": self.plate_text,
            "confidence": self.confidence,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "first_seen_time": self.first_seen_time,
            "last_seen_time": self.last_seen_time,
            "count": self.count,
            "best_detection": asdict(self.best_detection) if self.best_detection else None
        }
        
    @staticmethod
    def is_similar_plate(text1, text2):
        """Check if two plate texts are likely the same plate with OCR variations"""
        # If exact match, return True
        if text1 == text2:
            return True
            
        # Normalize both texts
        text1 = ''.join(c for c in text1.upper() if c.isalnum())
        text2 = ''.join(c for c in text2.upper() if c.isalnum())
        
        # If still exact match after normalization, return True
        if text1 == text2:
            return True
            
        # If length difference is too large, they're different plates
        if abs(len(text1) - len(text2)) > 2:
            return False
            
        # Calculate edit distance (Levenshtein distance)
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Calculate edit distance and determine similarity threshold
        distance = levenshtein(text1, text2)
        max_len = max(len(text1), len(text2))
        
        # If the texts have significant overlap or minimal edit distance, consider them the same
        # The threshold is more lenient for longer texts
        threshold = 0.25 if max_len >= 7 else 0.2
        return distance <= max(1, int(max_len * threshold))

class LicensePlateProcessor:
    """Dynamic license plate processing system"""
    
    def __init__(self):
        self.character_corrections = CHARACTER_CORRECTIONS
        self.license_patterns = LICENSE_PATTERNS
        self.ocr_engines = ocr_engines
        self.tesseract_available = tesseract_available
        
        # Configure preprocessing methods
        self.preprocessing_methods = [
            {'name': 'original', 'function': self._preprocess_original},
            {'name': 'adaptive_threshold', 'function': self._preprocess_adaptive_threshold},
            {'name': 'otsu_threshold', 'function': self._preprocess_otsu_threshold},
            {'name': 'contrast_enhanced', 'function': self._preprocess_contrast_enhanced},
            {'name': 'noise_removal', 'function': self._preprocess_noise_removal},
            {'name': 'edge_enhanced', 'function': self._preprocess_edge_enhanced}
        ]
        
    def _preprocess_original(self, img):
        """Basic preprocessing: resize and convert to grayscale"""
        if img is None or img.size == 0:
            return None
            
        try:
            # Resize the image
            h, w = img.shape[:2]
            target_height = 150
            aspect_ratio = w / h
            target_width = int(aspect_ratio * target_height)
            
            # Limit maximum width
            if target_width > 600:
                target_width = 600
                target_height = int(target_width / aspect_ratio)
                
            resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale if it's a color image
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
                
            return gray
        except Exception as e:
            print(f"Original preprocessing error: {e}")
            return img
    
    def _preprocess_adaptive_threshold(self, img):
        """Apply adaptive thresholding"""
        try:
            gray = self._preprocess_original(img)
            if gray is None:
                return None
                
            # Bilateral filter for noise removal while preserving edges
            denoised = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            return thresh
        except Exception as e:
            print(f"Adaptive threshold error: {e}")
            return self._preprocess_original(img)
    
    def _preprocess_otsu_threshold(self, img):
        """Apply Otsu's thresholding"""
        try:
            gray = self._preprocess_original(img)
            if gray is None:
                return None
                
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        except Exception as e:
            print(f"Otsu threshold error: {e}")
            return self._preprocess_original(img)
    
    def _preprocess_contrast_enhanced(self, img):
        """Enhance contrast using CLAHE"""
        try:
            gray = self._preprocess_original(img)
            if gray is None:
                return None
                
            # Create a CLAHE object
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            # Apply CLAHE
            enhanced = clahe.apply(gray)
            return enhanced
        except Exception as e:
            print(f"Contrast enhancement error: {e}")
            return self._preprocess_original(img)
    
    def _preprocess_noise_removal(self, img):
        """Remove noise using morphological operations"""
        try:
            thresh = self._preprocess_adaptive_threshold(img)
            if thresh is None:
                return None
                
            # Create a kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Apply morphological operations to remove noise
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            return closing
        except Exception as e:
            print(f"Noise removal error: {e}")
            return self._preprocess_original(img)
    
    def _preprocess_edge_enhanced(self, img):
        """Enhance edges in the image"""
        try:
            gray = self._preprocess_original(img)
            if gray is None:
                return None
                
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Detect edges using Canny
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate the edges
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Combine with original grayscale for a hybrid image
            result = cv2.addWeighted(gray, 0.8, dilated, 0.2, 0)
            
            return result
        except Exception as e:
            print(f"Edge enhancement error: {e}")
            return self._preprocess_original(img)
    
    def _read_with_easyocr(self, img, engine_key='primary'):
        """Read text using specified EasyOCR engine"""
        try:
            if img is None or img.size == 0:
                return None, 0.0
                
            if engine_key not in self.ocr_engines:
                engine_key = 'primary'
                
            results = self.ocr_engines[engine_key].readtext(img)
            
            if not results:
                return None, 0.0
                
            # Sort by confidence and get the best result
            results.sort(key=lambda x: x[2], reverse=True)
            text = results[0][1]
            confidence = results[0][2]
            
            return text, confidence
        except Exception as e:
            print(f"EasyOCR error ({engine_key}): {e}")
            return None, 0.0
    
    def _read_with_tesseract(self, img, psm=7):
        """Read text using Tesseract with configurable PSM"""
        if not self.tesseract_available:
            return None, 0.0
            
        try:
            if img is None or img.size == 0:
                return None, 0.0
                
            # Configure Tesseract
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Get text
            text = pytesseract.image_to_string(img, config=config).strip()
            
            # Get confidence if available (optional)
            try:
                data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                conf_values = [int(conf) for conf in data['conf'] if conf != '-1']
                confidence = sum(conf_values) / len(conf_values) / 100.0 if conf_values else 0.6
            except Exception:
                confidence = 0.6 if text else 0.0
                
            return text, confidence
        except Exception as e:
            print(f"Tesseract error (PSM {psm}): {e}")
            return None, 0.0
    
    def _segment_and_recognize_characters(self, img):
        """Segment the license plate into individual characters and recognize them"""
        try:
            if img is None or img.size == 0:
                return None, 0.0
                
            # Get grayscale image
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
                
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to find character-like shapes
            char_contours = []
            img_height = img.shape[0]
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Character-like aspect ratio and reasonable size
                if 0.1 < aspect_ratio < 1.0 and 0.4 * img_height < h < 0.9 * img_height:
                    char_contours.append((x, y, w, h))
            
            # Sort characters from left to right
            char_contours.sort(key=lambda c: c[0])
            
            # Need at least a few characters for a license plate
            if len(char_contours) < 3:
                return None, 0.0
                
            # Collect characters
            recognized_text = ""
            character_confidences = []
            
            for x, y, w, h in char_contours:
                # Extract the character with padding
                padding = 2
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(gray.shape[1], x + w + padding)
                y2 = min(gray.shape[0], y + h + padding)
                
                char_img = gray[y1:y2, x1:x2]
                
                if char_img.size == 0:
                    continue
                    
                # Resize to standard size for recognition
                char_img = cv2.resize(char_img, (30, 60))
                
                # Try to recognize the character
                char_text, char_conf = self._read_with_easyocr(char_img)
                
                if char_text and len(char_text) == 1 and char_text.isalnum():
                    recognized_text += char_text
                    character_confidences.append(char_conf)
            
            # Calculate overall confidence
            if character_confidences:
                avg_confidence = sum(character_confidences) / len(character_confidences)
            else:
                avg_confidence = 0.0
                
            return recognized_text, avg_confidence
        except Exception as e:
            print(f"Character segmentation error: {e}")
            return None, 0.0
    
    def _normalize_license_plate_text(self, text):
        """Remove extra characters and normalize the text"""
        if not text:
            return text
            
        # Remove spaces and convert to uppercase
        text = ''.join(text.split()).upper()
        
        # Remove non-alphanumeric characters except common delimiters
        text = re.sub(r'[^A-Z0-9-]', '', text)
        
        # Remove excessive delimiters
        text = re.sub(r'-+', '-', text)
        text = text.strip('-')
        
        return text
    
    def _generate_correction_candidates(self, text, country=None):
        """Generate potential corrections for the text based on common errors"""
        if not text:
            return []
            
        # Original text is always a candidate
        candidates = [text]
        
        # Apply character-by-character corrections
        corrected_variants = []
        
        # For each position in the text, try different corrections
        for i in range(len(text)):
            char = text[i]
            
            # Check if this character has potential corrections
            if char in self.character_corrections:
                for alternative in self.character_corrections[char]:
                    # Replace this character with the alternative
                    corrected = text[:i] + alternative + text[i+1:]
                    corrected_variants.append(corrected)
        
        candidates.extend(corrected_variants)
        
        # For specific countries, try known patterns
        if country and country in self.license_patterns.get('country_specific', {}):
            country_patterns = self.license_patterns['country_specific'][country]
            
            # Check for state prefixes for Indian plates
            if country == 'IN' and len(text) >= 2:
                state_prefixes = country_patterns.get('state_prefixes', [])
                
                # If the first two characters don't form a valid state code, 
                # but rest of the format seems right, try to correct it
                if text[:2] not in state_prefixes:
                    first_char_alternatives = self.character_corrections.get(text[0], [])
                    second_char_alternatives = self.character_corrections.get(text[1], [])
                    
                    # Try combinations of first and second character alternatives
                    for alt1 in [text[0]] + first_char_alternatives:
                        for alt2 in [text[1]] + second_char_alternatives:
                            potential_state = alt1 + alt2
                            if potential_state in state_prefixes:
                                corrected = potential_state + text[2:]
                                candidates.append(corrected)
                
                # Special case for Delhi plates (DL)
                if text[:2] in ['01', 'D1', '0L', 'CL', 'OL']:
                    candidates.append('DL' + text[2:])
        
        # Remove duplicates while preserving order
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _validate_license_plate(self, text, country=None):
        """Validate if the text matches known license plate patterns"""
        confidence = 0.0
        pattern_matched = False
        
        if not text or len(text) < 4:  # Too short to be valid
            return False, 0.0
        
        # Check country-specific patterns first if country is provided
        if country and country in self.license_patterns.get('country_specific', {}):
            country_data = self.license_patterns['country_specific'][country]
            patterns = country_data.get('patterns', [])
            base_confidence = country_data.get('confidence', 0.7)
            
            for pattern in patterns:
                if re.match(pattern, text):
                    confidence = base_confidence
                    pattern_matched = True
                    break
                    
            # Additional validation for specific countries
            if country == 'IN' and len(text) >= 2:
                state_prefixes = country_data.get('state_prefixes', [])
                if text[:2] in state_prefixes:
                    confidence += 0.1  # Bonus for valid state code
        
        # If no country-specific pattern matched, try generic patterns
        if not pattern_matched:
            generic_patterns = self.license_patterns.get('generic', {}).get('patterns', [])
            generic_confidence = self.license_patterns.get('generic', {}).get('confidence', 0.5)
            
            for pattern in generic_patterns:
                if re.match(pattern, text):
                    confidence = generic_confidence
                    pattern_matched = True
                    break
        
        # Additional checks for all plates
        if not pattern_matched:
            # Simple length and alphanumeric check
            if len(text) >= 5 and len(text) <= 10 and re.match(r'^[A-Z0-9-]+$', text):
                confidence = 0.3  # Minimal confidence
                pattern_matched = True
        
        return pattern_matched, confidence
    
    def _select_best_result(self, results, country=None):
        """Select the best result from multiple candidates"""
        if not results:
            return None, 0.0
            
        # Calculate scores for each result
        scored_results = []
        
        for result in results:
            text = result.text
            base_confidence = result.confidence
            
            # Normalize the text
            normalized_text = self._normalize_license_plate_text(text)
            
            # Validate against patterns
            valid, pattern_confidence = self._validate_license_plate(normalized_text, country)
            
            # Combine confidences - weight pattern matching highly
            combined_confidence = (base_confidence * 0.4) + (pattern_confidence * 0.6)
            
            # Add to scored results
            scored_results.append((normalized_text, combined_confidence, result))
        
        # Sort by combined confidence
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best result
        best_text, best_confidence, best_result = scored_results[0]
        
        return best_text, best_confidence
    
    def process_license_plate(self, img, country=None):
        """Process a license plate image using multiple techniques and select the best result"""
        if img is None or img.size == 0:
            return None, 0.0
            
        all_results = []
        
        # Apply various preprocessing methods and OCR
        for preproc_method in self.preprocessing_methods:
            try:
                preproc_name = preproc_method['name']
                preproc_func = preproc_method['function']
                
                # Apply preprocessing
                processed_img = preproc_func(img)
                
                if processed_img is None or processed_img.size == 0:
                    continue
                
                # Try with primary EasyOCR
                text, confidence = self._read_with_easyocr(processed_img, 'primary')
                if text:
                    all_results.append(OCRResult(text, confidence, 'easyocr_primary', preproc_name))
                
                # Try with Tesseract if available
                if self.tesseract_available:
                    for psm in [7, 8, 6]:  # Different Tesseract Page Segmentation Modes
                        text, confidence = self._read_with_tesseract(processed_img, psm)
                        if text:
                            all_results.append(OCRResult(text, confidence, f'tesseract_psm{psm}', preproc_name))
                
                # Try character segmentation if other methods have low confidence
                if not any(r.confidence > 0.7 for r in all_results):
                    text, confidence = self._segment_and_recognize_characters(processed_img)
                    if text:
                        all_results.append(OCRResult(text, confidence, 'char_segmentation', preproc_name))
                
            except Exception as e:
                print(f"Error in processing method {preproc_method['name']}: {e}")
                continue
        
        # Get correction candidates for each result
        expanded_results = []
        for result in all_results:
            # Add the original result
            expanded_results.append(result)
            
            # Generate correction candidates
            candidates = self._generate_correction_candidates(result.text, country)
            
            # Add correction candidates (excluding the original text)
            for candidate in candidates:
                if candidate != result.text:
                    # Slightly lower confidence for corrections
                    correction_confidence = result.confidence * 0.95
                    expanded_results.append(
                        OCRResult(candidate, correction_confidence, f"{result.method}_corrected", result.preprocessing)
                    )
        
        # Select the best result
        if expanded_results:
            best_text, best_confidence = self._select_best_result(expanded_results, country)
            return best_text, best_confidence
            
        return None, 0.0

class VideoProcessor:
    """Process videos to detect and track license plates"""
    
    def __init__(self, country=None, sample_rate=FRAME_SAMPLE_RATE, max_frames=MAX_FRAMES_TO_PROCESS):
        self.country = country
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.plate_processor = LicensePlateProcessor()
        self.plate_trackers = {}  # Dictionary to track unique license plates
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        # Box format: [x1, y1, x2, y2]
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def process_frame(self, frame, frame_number, timestamp):
        """Process a single video frame to detect license plates"""
        if frame is None or not models_loaded:
            return []
        
        try:
            # Make a copy of the frame
            result_frame = frame.copy()
            
            # First detect vehicles
            vehicle_detections = vehicle_detector(frame)[0]
            
            # Then detect license plates
            license_detections = license_plate_detector(frame)[0]
            
            # Process each license plate
            license_plates = license_detections.boxes.data.tolist()
            vehicle_boxes = vehicle_detections.boxes.data.tolist() if len(vehicle_detections.boxes) > 0 else []
            
            plate_detections = []
            
            for license_plate in license_plates:
                try:
                    x1, y1, x2, y2, conf, _ = map(float, license_plate)
                    
                    # Skip low confidence detections
                    if conf < 0.4:
                        continue
                        
                    # Ensure coordinates are integers and within image bounds
                    x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)])
                    
                    # Skip invalid regions
                    if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 100:  # Too small
                        continue
                    
                    # Add padding to license plate
                    padding_x = int((x2 - x1) * 0.05)
                    padding_y = int((y2 - y1) * 0.1)
                    pad_x1 = max(0, x1 - padding_x)
                    pad_y1 = max(0, y1 - padding_y)
                    pad_x2 = min(frame.shape[1], x2 + padding_x)
                    pad_y2 = min(frame.shape[0], y2 + padding_y)
                    
                    # Crop license plate region
                    plate_crop = frame[pad_y1:pad_y2, pad_x1:pad_x2]
                    
                    # Skip if crop is empty
                    if plate_crop is None or plate_crop.size == 0:
                        continue
                    
                    # Process license plate
                    plate_text, text_confidence = self.plate_processor.process_license_plate(plate_crop, self.country)
                    
                    # If text was found and confidence is sufficient
                    if plate_text and text_confidence > MIN_DETECTION_CONFIDENCE:
                        # Find associated vehicle (if any)
                        vehicle_bbox = None
                        for vehicle in vehicle_boxes:
                            v_x1, v_y1, v_x2, v_y2, v_conf, _ = map(float, vehicle)
                            # Check if license plate is inside vehicle bounding box
                            if v_x1 <= x1 and v_y1 <= y1 and v_x2 >= x2 and v_y2 >= y2:
                                vehicle_bbox = [int(v_x1), int(v_y1), int(v_x2), int(v_y2)]
                                break
                        
                        # Create detection object
                        detection = PlateDetection(
                            plate_text=plate_text,
                            confidence=text_confidence,
                            bbox=[x1, y1, x2, y2],
                            frame_number=frame_number,
                            timestamp=timestamp,
                            vehicle_bbox=vehicle_bbox
                        )
                        
                        # Add to detections
                        plate_detections.append(detection)
                        
                        # Draw on result frame
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add text with background
                        font_scale = 0.9
                        text_thickness = 2
                        (text_width, text_height), _ = cv2.getTextSize(
                            plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
                        )
                        
                        if y1 > text_height + 10:
                            text_x, text_y = x1, y1 - 10
                        else:
                            text_x, text_y = x1, y2 + text_height + 10
                            
                        # Draw text background
                        cv2.rectangle(
                            result_frame, 
                            (text_x, text_y - text_height - 5), 
                            (text_x + text_width + 60, text_y + 5), 
                            (0, 0, 0), 
                            -1
                        )
                        
                        # Draw plate text
                        cv2.putText(
                            result_frame, 
                            plate_text, 
                            (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, 
                            (0, 255, 0), 
                            text_thickness
                        )
                        
                        # Add confidence
                        conf_text = f"{int(text_confidence * 100)}%"
                        cv2.putText(
                            result_frame,
                            conf_text,
                            (text_x + text_width + 5, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            1
                        )
                except Exception as e:
                    print(f"Error processing license plate: {e}")
                    continue
            
            return plate_detections, result_frame
                    
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            traceback.print_exc()
            return [], frame
    
    def update_trackers(self, detections):
        """Update license plate trackers with new detections"""
        for detection in detections:
            plate_text = detection.plate_text
            
            # Check if this plate is similar to any existing tracked plate
            matched = False
            for tracker_id, tracker in self.plate_trackers.items():
                if PlateTracker.is_similar_plate(tracker.plate_text, plate_text):
                    # Update existing tracker
                    tracker.update(detection)
                    matched = True
                    break
            
            # If no match found, create a new tracker
            if not matched:
                tracker_id = str(uuid.uuid4())
                self.plate_trackers[tracker_id] = PlateTracker(
                    plate_id=tracker_id,
                    plate_text=plate_text,
                    first_seen_frame=detection.frame_number,
                    last_seen_frame=detection.frame_number,
                    first_seen_time=detection.timestamp,
                    last_seen_time=detection.timestamp,
                    detections=[detection],
                    confidence=detection.confidence,
                    best_detection=detection,
                    count=1
                )
    
    async def process_video(self, video_path):
        """Process a video file to detect license plates"""
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Create results directory
            timestamp = int(time.time())
            results_dir = os.path.join(TEMP_DIR, f"video_results_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Create directory for frame images
            frames_dir = os.path.join(results_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Initialize variables
            frame_number = 0
            processed_frames = 0
            saved_frames = []
            video_info = {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "processed_frames": 0,
                "plates_detected": 0
            }
            
            # Clear existing trackers
            self.plate_trackers = {}
            
            # Process frames
            while cap.isOpened() and processed_frames < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every N-th frame to save time
                if frame_number % self.sample_rate == 0:
                    # Calculate timestamp for this frame
                    timestamp = frame_number / fps if fps > 0 else 0
                    
                    # Process frame
                    detections, result_frame = self.process_frame(frame, frame_number, timestamp)
                    
                    # Update trackers with new detections
                    self.update_trackers(detections)
                    
                    # Save frame with detections
                    if detections:
                        frame_path = os.path.join(frames_dir, f"frame_{frame_number:06d}.jpg")
                        cv2.imwrite(frame_path, result_frame)
                        saved_frames.append({
                            "frame_number": frame_number,
                            "timestamp": timestamp,
                            "path": frame_path,
                            "detections": [d.to_dict() for d in detections]
                        })
                    
                    processed_frames += 1
                    
                    # Allow other tasks to run (important for FastAPI server)
                    if processed_frames % 10 == 0:
                        await asyncio.sleep(0.01)
                
                frame_number += 1
            
            # Close video
            cap.release()
            
            # Create summary report
            plates_summary = [tracker.to_dict() for tracker in self.plate_trackers.values()]
            
            # Update video info
            video_info["processed_frames"] = processed_frames
            video_info["plates_detected"] = len(self.plate_trackers)
            
            # Save results
            results = {
                "video_info": video_info,
                "plates": plates_summary,
                "frames": saved_frames
            }
            
            # Save JSON results
            results_path = os.path.join(results_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            
            return results, results_dir
            
        except Exception as e:
            print(f"Error processing video: {e}")
            traceback.print_exc()
            return {"error": str(e)}, None
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

def process_image(image: np.ndarray, country: str = '') -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Process a single image to detect license plates"""
    if not models_loaded or image is None:
        return [], image
        
    try:
        # Create processor objects
        plate_processor = LicensePlateProcessor()
        
        # Make a copy of the image
        result_image = image.copy()
        
        # Detect license plates
        license_detections = license_plate_detector(image)[0]
        
        detected_plates = []
        license_plates = license_detections.boxes.data.tolist()
        
        for license_plate in license_plates:
            try:
                x1, y1, x2, y2, conf, _ = map(float, license_plate)
                
                # Skip low confidence detections
                if conf < 0.4:
                    continue
                    
                # Ensure coordinates are integers and within image bounds
                x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)])
                
                # Skip invalid regions
                if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 100:  # Too small
                    continue
                
                # Add padding to license plate
                padding_x = int((x2 - x1) * 0.05)
                padding_y = int((y2 - y1) * 0.1)
                pad_x1 = max(0, x1 - padding_x)
                pad_y1 = max(0, y1 - padding_y)
                pad_x2 = min(image.shape[1], x2 + padding_x)
                pad_y2 = min(image.shape[0], y2 + padding_y)
                
                # Crop license plate region
                plate_crop = image[pad_y1:pad_y2, pad_x1:pad_x2]
                
                # Skip if crop is empty
                if plate_crop is None or plate_crop.size == 0:
                    continue
                
                # Process license plate
                plate_text, text_confidence = plate_processor.process_license_plate(plate_crop, country)
                
                # If text was found and confidence is sufficient
                if plate_text and text_confidence > MIN_DETECTION_CONFIDENCE:
                    detected_plates.append({
                        "text": plate_text, 
                        "confidence": float(text_confidence),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
                    
                    # Draw rectangle around plate
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add text on image with good visibility
                    font_scale = 0.9
                    text_thickness = 2
                    
                    # Calculate text size for background
                    (text_width, text_height), _ = cv2.getTextSize(
                        plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
                    )
                    
                    # Position text above or below the plate depending on space
                    if y1 > text_height + 10:
                        text_x = x1
                        text_y = y1 - 10
                        rect_y1 = text_y - text_height - 5
                        rect_y2 = text_y + 5
                    else:
                        text_x = x1
                        text_y = y2 + text_height + 10
                        rect_y1 = text_y - text_height - 5
                        rect_y2 = text_y + 5
                    
                    # Draw semi-transparent background for text
                    cv2.rectangle(
                        result_image, 
                        (text_x, rect_y1), 
                        (text_x + text_width + 60, rect_y2), 
                        (0, 0, 0), 
                        -1
                    )
                    
                    # Draw the text in bright green
                    cv2.putText(
                        result_image, 
                        plate_text, 
                        (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        (0, 255, 0), 
                        text_thickness
                    )
                    
                    # Add confidence percentage
                    conf_text = f"{int(text_confidence * 100)}%"
                    cv2.putText(
                        result_image,
                        conf_text,
                        (text_x + text_width + 5, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1
                    )
            except Exception as e:
                print(f"Error processing individual plate: {e}")
                continue
                
    except Exception as e:
        print(f"Image processing error: {e}")
        traceback.print_exc()
        return [], image
        
    return detected_plates, result_image

@app.post("/api/detect_license_plate/")
async def detect_license_plate(file: UploadFile = File(...), country: str = Form("", description="Country code (e.g., US, IN, UK)")):
    """
    Process an uploaded image to detect and read license plates.
    
    - **file**: Image file (JPEG, PNG)
    - **country**: Country code to optimize OCR for specific license plate formats
    """
    print(f"Processing image with country: {country}")
    
    # Read uploaded file
    try:
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400, 
                content={"error": "Invalid image file"}
            )

        # Check image dimensions
        h, w = image.shape[:2]
        max_dim = 1600  # Limit image size to prevent memory issues
        
        if max(h, w) > max_dim:
            # Resize image if too large
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            print(f"Image resized from {h}x{w} to {image.shape[0]}x{image.shape[1]}")

        # Process image
        print("Starting image processing")
        detected_plates, result_image = process_image(image, country)
        print(f"Processing complete. Detected {len(detected_plates)} plates.")

        # Encode the result image to base64 for JSON response
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_bytes = buffer.tobytes()
        result_image_base64 = base64.b64encode(result_image_bytes).decode('utf-8')

        # Response data
        response_data = {
            "plates": detected_plates,
            "image_base64": result_image_base64
        }

        # Save to Firestore if enabled
        if firestore_enabled and detected_plates:
            try:
                entry_id = str(uuid.uuid4())
                firestore_data = {
                    "id": entry_id,
                    "plates": detected_plates,
                    "country": country,
                    "timestamp": firestore.SERVER_TIMESTAMP
                }
                db.collection("license_plates").document(entry_id).set(firestore_data)
                response_data["id"] = entry_id
            except Exception as e:
                print(f"Firestore error: {e}")

        return JSONResponse(
            status_code=200,
            content=response_data
        )
    except Exception as e:
        print(f"Request processing error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

@app.post("/api/process_video/")
async def process_video_endpoint(
    file: UploadFile = File(...), 
    country: str = Form("", description="Country code (e.g., US, IN, UK)"),
    sample_rate: int = Form(FRAME_SAMPLE_RATE, description="Process every N-th frame"),
    max_frames: int = Form(MAX_FRAMES_TO_PROCESS, description="Maximum frames to process")
):
    """
    Process a video file to detect and track license plates.
    
    - **file**: Video file (MP4, AVI)
    - **country**: Country code to optimize OCR for specific license plate formats
    - **sample_rate**: Process every N-th frame (default: 5)
    - **max_frames**: Maximum number of frames to process (default: 300)
    
    Returns a summary of detected license plates and links to result frames.
    """
    print(f"Processing video with country: {country}, sample_rate: {sample_rate}, max_frames: {max_frames}")
    
    try:
        # Validate file
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file format. Please upload a video file (MP4, AVI, MOV, MKV)."}
            )
        
        # Save uploaded file to temporary location
        temp_file = os.path.join(TEMP_DIR, f"upload_{int(time.time())}_{file.filename}")
        with open(temp_file, "wb") as f:
            f.write(await file.read())
        
        # Process video
        processor = VideoProcessor(country, sample_rate, max_frames)
        results, results_dir = await processor.process_video(temp_file)
        
        if "error" in results:
            return JSONResponse(
                status_code=500,
                content={"error": results["error"]}
            )
        
        # Create summary response
        summary = {
            "video_info": results["video_info"],
            "plates_detected": len(results["plates"]),
            "unique_plates": [
                {
                    "plate_text": plate["plate_text"],
                    "confidence": plate["confidence"],
                    "first_seen": plate["first_seen_time"],
                    "count": plate["count"]
                }
                for plate in results["plates"]
            ],
            # Include paths to a few sample frames
            "sample_frames": [
                {
                    "frame_number": frame["frame_number"],
                    "timestamp": frame["timestamp"],
                    "path": os.path.basename(frame["path"])
                }
                for frame in results["frames"][:5]  # Just include first 5 frames
            ],
            "results_dir": os.path.basename(results_dir)
        }
        
        # Save to Firestore if enabled
        if firestore_enabled and results["plates"]:
            try:
                entry_id = str(uuid.uuid4())
                firestore_data = {
                    "id": entry_id,
                    "summary": summary,
                    "country": country,
                    "timestamp": firestore.SERVER_TIMESTAMP
                }
                db.collection("video_processing").document(entry_id).set(firestore_data)
                summary["id"] = entry_id
            except Exception as e:
                print(f"Firestore error: {e}")
        
        return JSONResponse(
            status_code=200,
            content=summary
        )
        
    except Exception as e:
        print(f"Video processing error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Video processing failed: {str(e)}"}
        )
    finally:
        # Clean up temporary file
        if 'temp_file' in locals() and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

@app.get("/api/video_results/{result_dir}/{frame_file}")
async def get_video_frame(result_dir: str, frame_file: str):
    """
    Get a processed video frame.
    
    - **result_dir**: Results directory name
    - **frame_file**: Frame file name
    """
    try:
        frame_path = os.path.join(TEMP_DIR, result_dir, "frames", frame_file)
        if not os.path.exists(frame_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Frame not found"}
            )
        
        # Return the frame image
        with open(frame_path, "rb") as f:
            image_bytes = f.read()
        
        return StreamingResponse(BytesIO(image_bytes), media_type="image/jpeg")
    except Exception as e:
        print(f"Error retrieving frame: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve frame: {str(e)}"}
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify the backend is running."""
    status = {
        "status": "healthy",
        "models_loaded": models_loaded,
        "firestore_enabled": firestore_enabled,
        "tesseract_available": tesseract_available,
        "ocr_engines_available": list(ocr_engines.keys())
    }
    return JSONResponse(status_code=200, content=status)

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)