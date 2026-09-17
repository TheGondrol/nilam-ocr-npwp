import numpy as np
from dotenv import load_dotenv
import cv2
import re
import os
import time
import json
from rapidfuzz import fuzz, process
import statistics
import mediapipe as mp
import io
from paddleocr import PaddleOCR
# from ppocr.utils.logging import get_logger
import logging
from PIL import Image
from utils.prefix import mapping_pekerjaan, mapping_status, mapping_agama, mapping_jeniskelamin, symbol_mapping, alphabet_mapping, digit_mapping, bulan_dict, keydata, keyid, message_blur, message_glare, message_rotated, message_median
# from pymilvus import model, MilvusClient

load_dotenv()
THRESHOLD_RATIO = float(os.environ["THRESHOLD_RATIO"])
THRESHOLD_PARTIAL = float(os.environ["THRESHOLD_PARTIAL"])
THRESHOLD_CONFIDENCE = float(os.environ["THRESHOLD_CONFIDENCE"])
THRESHOLD_BLUR = float(os.environ["THRESHOLD_BLUR"])
THRESHOLD_MEDIAN = float(os.environ["THRESHOLD_MEDIAN"])
THRESHOLD_MIN_AREA_GLARE=int(os.environ["THRESHOLD_MIN_AREA_GLARE"])
GLARE_PADDING_SIZE=int(os.environ["GLARE_PADDING_SIZE"])

# Configure logger
# logger = get_logger()
# logger.setLevel(logging.ERROR)

# Initialize OCR
_ocr_instance = None

def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=True, 
            use_doc_unwarping=True, 
            use_textline_orientation=True,
            )
    return _ocr_instance

#untuk image quality
def image_quality(image_bytes: bytes, ocr_result: list) -> tuple:
    """
    Mengecek kualitas gambar berdasarkan blur dan glare.

    Args:
        image_bytes (bytes): Gambar dalam bentuk bytes.

    Returns:
        tuple: (is_blurry, is_glare), masing-masing bertipe bool.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)
    image_cv = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    is_blurry, _ = blur_detection(image_cv, THRESHOLD_BLUR)
    is_glare = detect_glare_with_text_analysis(np_image, ocr_result)
    is_rotated = detect_image_rotation(np_image)
    return is_blurry, is_glare, is_rotated

def is_landscape(image):
    """Check if image is in landscape orientation (width > height)"""
    height, width = image.shape[:2]
    return width > height

def is_face_in_box(face_x, face_y, face_width, face_height, box):
    """
    Check if face is completely inside the specified box
    box is defined as (x, y, width, height)
    Returns True if face is completely inside the box, False otherwise
    """
    box_x, box_y, box_width, box_height = box
    
    # Check if face is completely inside the box
    if (face_x >= box_x and
        face_y >= box_y and
        face_x + face_width <= box_x + box_width and
        face_y + face_height <= box_y + box_height):
        return True
    else:
        return False

def rotated_detection(image, min_face_proportion=0.2):
    """

    Process an image according to the requirements.
    Returns: (is_accepted, reason, annotated_image)
    """
    # Read image
    # image = cv2.imread(np_image)
    if image is None:
        return False, "Error: Could not read image", None
    
    # Check if image is landscape
    if not is_landscape(image):
        return False, "Rejected: Image is not in landscape orientation", image
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Configure face detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # 1 for full-range detection
        min_detection_confidence=0.5
    )
    
    # Convert to RGB for MediaPipe
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = face_detection.process(image)
    
    # Create a copy for visualization
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    
    # Define the verification box (in the right side of the image)
    box_width = int(width * 0.3)  # 30% of image width
    box_height = int(height * 0.6)  # 60% of image height
    box_x = width - box_width - int(width * 0.02)  # Positioned on the right with 5% margin
    box_y = int(height * 0.44) - (box_height // 2)  # Centered vertically
    verification_box = (box_x, box_y, box_width, box_height)
    
    # Calculate verification box area
    box_area = box_width * box_height
    
    # Draw the verification box
    cv2.rectangle(
        annotated_image, 
        (box_x, box_y), 
        (box_x + box_width, box_y + box_height), 
        (0, 255, 0),  # Green color
        2
    )
    
    # Add verification box label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        annotated_image, 
        "Face Position Area", 
        (box_x, box_y - 10), 
        font, 0.7, 
        (0, 255, 0),  # Green color
        2
    )
    
    # Check if any faces were detected
    if not results.detections:
        return False, "No face detected", annotated_image
    
    # Process detected faces
    for detection in results.detections:
        # Draw face detection
        mp_drawing.draw_detection(annotated_image, detection)
        
        # Get bounding box
        bboxC = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute pixel values
        face_x = int(bboxC.xmin * width)
        face_y = int(bboxC.ymin * height)
        face_w = int(bboxC.width * width)
        face_h = int(bboxC.height * height)
        
        # Calculate face area
        face_area = face_w * face_h
        
        # Calculate proportion
        face_box_proportion = face_area / box_area
        
        # Print area information
        print(f"Face area: {face_area} pixels")
        print(f"Box area: {box_area} pixels")
        print(f"Face to box proportion: {face_box_proportion:.4f}")
        
        # Check if face is entirely inside the verification box
        if is_face_in_box(face_x, face_y, face_w, face_h, verification_box):
            # Check if face has minimum required proportion
            if face_box_proportion < min_face_proportion:
                # Add face position info to annotation
                cv2.putText(
                    annotated_image, 
                    f"Face too small: {face_box_proportion:.2f}", 
                    (face_x, face_y - 10), 
                    font, 0.7, 
                    (0, 0, 255),  # Red color
                    2
                )
                return False, f"Face is positioned correctly but too small (proportion: {face_box_proportion:.4f}, min required: {min_face_proportion})", annotated_image
            else:
                # Add face position info to annotation
                cv2.putText(
                    annotated_image, 
                    f"Face in position: {face_box_proportion:.2f}", 
                    (face_x, face_y - 10), 
                    font, 0.7, 
                    (0, 255, 0),  # Green color
                    2
                )
                return True, f"Face is properly positioned with good proportion ({face_box_proportion:.4f})", annotated_image
        else:
            # Add face position info to annotation
            cv2.putText(
                annotated_image, 
                "Face out of position", 
                (face_x, face_y - 10), 
                font, 0.7, 
                (0, 0, 255),  # Red color
                2
            )
    
    # If we get here, no face was properly positioned
    return False, "Face not properly positioned in the verification area", annotated_image

def detect_image_rotation(image):
    # ===== CONFIGURATION (EDIT THESE VALUES) =====
    
    # Set to True if you want to save the output image
    save_output = False
    
    # Path where to save the output image (only used if save_output is True)
    output_path = "result.jpg"
    
    # Minimum required face to box proportion (adjust as needed)
    min_face_proportion = 0.12  # Face must occupy at least 15% of the verification box
    
    # ===== END OF CONFIGURATION =====
    
    # Process the image
    is_accepted, reason, annotated_image = rotated_detection(
        image,
        min_face_proportion=min_face_proportion
    )
    
    # Print the result
    print(f"Result: {reason}")
    print(f"Image is {'ACCEPTED' if is_accepted else 'REJECTED'}")
    
    # Save output if requested
    if save_output and annotated_image is not None:
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")
    return not is_accepted

def calculate_median_brightness(img):
    """
    Calculate the median brightness of an image using HSV.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Image in BGR format
        
    Returns:
    --------
    float
        Median brightness value (0-255)
    """
    # Convert the image from BGR to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Extract the V channel (brightness)
    v_channel = hsv_img[:, :, 2]
    
    # Calculate the median brightness
    median_brightness = np.median(v_channel)
    
    return median_brightness

def determine_glare_threshold(median_brightness):
    """
    Determine appropriate glare threshold based on median brightness.
    
    Parameters:
    -----------
    median_brightness : float
        Median brightness value of the image (0-255)
        
    Returns:
    --------
    int
        Threshold value for glare detection
    """
    if median_brightness > 240:
        # For very bright images, use higher threshold 
        # but cap at 253 or median + 3, whichever is smaller
        return min(254, int(median_brightness) + 5)
    elif 220 <= median_brightness <= 240:
        return 250
    elif 200 <= median_brightness < 220:
        return 240
    elif 120 <= median_brightness < 200:
        return 220
    else:  # median_brightness < 120
        return 150
    
def detect_glare_with_text_analysis(np_image, ocr_result):
    """
    Optimized glare detection with text analysis and adaptive thresholding.
    """
    
    start_time = time.time()
    # Konversi RGB ke BGR (karena OpenCV pakai BGR)
    img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]
    
    # Resize large images to improve speed (adjust max_dimension as needed)
    # max_dimension = 1200
    # if max(height, width) > max_dimension:
    #     scale = max_dimension / max(height, width)
    #     img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Calculate median brightness and determine threshold
    median_brightness = calculate_median_brightness(img)
    threshold_value = determine_glare_threshold(median_brightness)
    
    print(f"Image median brightness: {median_brightness:.2f}")
    print(f"Using adaptive glare threshold: {threshold_value}")
    
    # Convert directly to V channel without creating full HSV
    # This is faster than converting to full HSV
    bgr_channels = cv2.split(img)
    v_channel = cv2.max(cv2.max(bgr_channels[0], bgr_channels[1]), bgr_channels[2])
    
    # Threshold the V channel (brightness)
    _, glare_mask = cv2.threshold(v_channel, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Clean up noise with optimized morphological operations
    kernel = np.ones((5, 5), np.uint8)
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of glare regions with simplified chain approximation
    contours, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area more efficiently
    significant_contours = []
    glare_areas = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > THRESHOLD_MIN_AREA_GLARE:
            significant_contours.append(cnt)
            glare_areas.append(area)
    
    # Sort only if needed
    if len(significant_contours) > 3:
        # Sort only the top 3 by area for padding
        indices = np.argsort(glare_areas)[-3:][::-1]
        top_contours = [significant_contours[i] for i in indices]
    else:
        top_contours = significant_contours
    
    # Create padded glare mask more efficiently
    padded_glare_mask = np.zeros_like(glare_mask)
    
    # First, fill all original glare contours
    cv2.drawContours(padded_glare_mask, significant_contours, -1, 255, -1)
    
    # Add padding to only the largest contours
    if GLARE_PADDING_SIZE > 0:
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (2 * GLARE_PADDING_SIZE + 1, 2 * GLARE_PADDING_SIZE + 1)
        )
        
        for cnt in top_contours:
            # Create mask for this contour
            mask_for_dilation = np.zeros_like(glare_mask)
            cv2.drawContours(mask_for_dilation, [cnt], 0, 255, -1)
            
            # Dilate to create padding
            dilated_mask = cv2.dilate(mask_for_dilation, dilation_kernel, iterations=1)
            
            # Add to padding mask
            padded_glare_mask = cv2.bitwise_or(padded_glare_mask, dilated_mask)
    
    # Process OCR results
    text_regions = []
    if ocr_result is not None:
        for line in ocr_result:
            box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = line[1][0]
            confidence = line[1][1]
            
            # Skip low confidence text to reduce noise
            if confidence < 0.6:
                continue
                
            # Convert box to contour format
            box_np = np.array(box, dtype=np.int32)
            text_regions.append({
                'contour': box_np,
                'text': text,
                'confidence': confidence
            })
    
    # Analyze intersection between glare and text regions
    affected_regions = []
    
    # Use vectorized operations for masks when possible
    for text_region in text_regions:
        # Create binary mask for text region
        text_mask = np.zeros(padded_glare_mask.shape, dtype=np.uint8)
        cv2.fillPoly(text_mask, [text_region['contour']], 255)
        
        # Calculate intersection quickly
        intersection = cv2.bitwise_and(text_mask, padded_glare_mask)
        
        # Calculate areas
        text_area = cv2.countNonZero(text_mask)
        intersection_area = cv2.countNonZero(intersection)
        
        # Calculate percentage of text affected by glare
        if text_area > 0:
            affected_percentage = (intersection_area / text_area) * 100
        else:
            affected_percentage = 0
        
        # Consider text affected if more than 5% of its area is covered by glare
        if affected_percentage > 5:
            affected_regions.append({
                'text': text_region['text'],
                'affected_percentage': affected_percentage
            })
    
    # Calculate overall impact metrics
    affected_text_count = len(affected_regions)
    text_regions_count = len(text_regions)
    
    if text_regions_count > 0:
        affected_text_percentage = (affected_text_count / text_regions_count) * 100
    else:
        affected_text_percentage = 0
    
    # Determine overall impact level
    if affected_text_percentage > 30:
        overall_impact = "High"
    elif affected_text_percentage > 10:
        overall_impact = "Medium"
    else:
        overall_impact = "Low"
    
    # Make final decision: reject if any text is affected, accept otherwise
    decision = True if affected_text_count > 0 else False
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Prepare result
    result = {
        'glare_count': len(significant_contours),
        'text_regions_count': text_regions_count,
        'affected_text_count': affected_text_count,
        'affected_text_percentage': affected_text_percentage,
        'affected_regions': affected_regions,
        'overall_readability_impact': overall_impact,
        'decision': decision,
        'threshold_used': threshold_value,
        'median_brightness': median_brightness,
    }
    print(result)
    
    return affected_text_count > 0

def convert_ocr_image(image:bytes) -> list:
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    image.seek(0)

    # Convert PIL Image to numpy array (required by PaddleOCR)
    image_np = np.asarray(image)
    return image_np

##untuk ocr
def perform_ocr(image_bytes: bytes) -> list:
    """
    Melakukan OCR pada gambar dan mengembalikan hasil deteksi teks beserta skor confidence.

    Args:
        image_bytes (bytes): Gambar dalam bentuk bytes.

    Returns:
        list: List hasil OCR, setiap elemen berupa tuple (text, confidence).
    """
    ocr = get_ocr()

    # Convert Image to numpy array (required by PaddleOCR)
    image_np = convert_ocr_image(image_bytes)
    results = ocr.predict(image_np)
    results = results[0]
    # Assume 'result' is your OCR output dictionary
    rec_texts = results['rec_texts']
    rec_scores = results['rec_scores']
    rec_polys = results['rec_polys']

    # Combine into desired format
    output = [
        (poly.tolist(),(text, float(score)))
        for text, score, poly in zip(rec_texts, rec_scores, rec_polys)
    ]

    height, width = image_np.shape[:2]
    if output[0] is None or len(output) == 0:
        return None
    result = filter_result(output, width)
    print("Hasil OCR:", result)
    return result

def filter_result(result, width):
    threshold = (width/5)*4
    filtered_line = []
    for line in result:
        cordinate = line[0]
        # print("Cordinate:", cordinate)
        if all(float(c[0]) <= threshold for c in cordinate):  # semua titik X harus <= 550
            filtered_line.append(line)
    return filtered_line

def get_coorthres(result):
    most_right = 0
    for line in result:
        if max(line[0]) > most_right:
            most_right = max(line[0])
        else:
            continue
    return most_right

#check median confidence score dari hasil ocr
def check_confidencemedian(results: list) -> bool:
    """
    Mengecek apakah median confidence hasil OCR memenuhi threshold.

    Args:
        results (list): List hasil OCR, setiap elemen tuple (text, confidence).

    Returns:
        bool: True jika median confidence >= THRESHOLD_MEDIAN, False jika tidak.
    """
    if not results:
        return 0
    results = [line[1] for line in results]
    scores = [score for _, score in results if isinstance(score, (int, float))]
    if not scores:
        return 0
    median_score = statistics.median(scores)
    print("median score", median_score)
    return median_score <= THRESHOLD_MEDIAN


def fuzz_ratio(query, data):
    """
    Menghitung skor kemiripan (similarity score) antara dua string
    menggunakan metode rasio fuzzy (fuzz.ratio) dari RapidFuzz.

    Args:
        query (str): String pertama yang akan dibandingkan.
        data (str): String kedua yang akan dibandingkan.

    Returns:
        int: Skor kemiripan antara 0-100, semakin tinggi semakin mirip.
    """
    score = fuzz.ratio(query.lower(), data.lower())
    return score

def fuzz_partial(query, data):
    """
    Menghitung skor kemiripan (similarity score) antara dua string
    menggunakan metode partial_ratio dari RapidFuzz.

    Args:
        query (str): String pertama yang akan dibandingkan.
        data (str): String kedua yang akan dibandingkan.

    Returns:
        int: Skor kemiripan antara 0-100, semakin tinggi semakin mirip.
    """
    score = fuzz.partial_ratio(query.lower(), data.lower())
    return score

def blur_detection(grayimage, threshold=100, show_plot=True):
    """
    Detect blur in an image using the Laplacian variance method.
    
    Parameters:
    - image_path: Path to the local image file or PIL Image object
    - threshold: Variance threshold below which image is considered blurry
    - show_plot: Whether to display visualization plots (default: True)
    
    Returns:
    - Boolean: True if image is blurry (variance below threshold)
    - Float: Variance score
    """
    
    # Compute the Laplacian
    laplacian = cv2.Laplacian(grayimage, cv2.CV_64F)
    
    # Calculate variance
    variance = laplacian.var()

    print(f"Variance score: {variance}")
    
    # Determine if the image is blurry
    is_blurry = variance < threshold

    # Return blur status and variance score
    return is_blurry, variance

def kelompokkan_meet_ttl(parts: list) -> tuple:
    if len(parts) >= 4:
        tempat = ' '.join(parts[:-3])
        day = parts[-3]
        month_raw = parts[-2]
        year = parts[-1]
    else:
        # Fallback jika parts kurang dari 4 elemen
        tempat = parts[0] if len(parts) > 0 else None
        day = parts[1][:2] if len(parts) > 1 else None
        month_raw = parts[2].lower() if len(parts) > 2 else None
        year = parts[3] if len(parts) > 3 else None
    return tempat, day, month_raw, year

def clean_tempat_meet_ttl(tempat:str)->str:
    try:
        tempat = re.sub(r"^[^A-Za-z]+", "", tempat)  # buang karakter non-huruf dari awal
        colon_match = re.search(r"[：:]\s*(\S+)", tempat)
        if colon_match:
            tempat = colon_match.group(1)
        tempat = correct_digits_to_alphabets(tempat)
        match = re.match(r"([A-Za-z\s]+)", tempat)
        if match:
            tempat_clear = True
            tempat = match.group(1)
            print(f"Tempat terpisah: {tempat} {tempat_clear}")
        return tempat
    except Exception as e:
        print(f"Error cleaning tempat meet ttl : {e}")
        return tempat

def meet_tgllength(parts: list) -> tuple:
    """
    Mengekstrak tempat, hari, bulan, dan tahun dari list hasil split tanggal lahir.

    Args:
        parts (list): List hasil split string tanggal lahir.

    Returns:
        tuple: (tempat, day, month, year) jika berhasil, jika gagal semuanya None.
    """
    try:
        # Normalisasi panjang parts
        tempat, day, month_raw, year = kelompokkan_meet_ttl(parts)

        # Konversi bulan jika perlu
        month = bulan_dict.get(month_raw, month_raw.zfill(2) if month_raw and month_raw.isdigit() else None)

        # Jika tempat mengandung digit, pisahkan
        print(f"Ekstrak tempat: {tempat}, day: {day}, month_raw: {month_raw}, year: {year}")
        tempat_clear = False
        if tempat:
            tempat = clean_tempat_meet_ttl(tempat)
        if day and len(day) > 2:
            match = re.match(r"([A-Za-z]+)(\d+)", day)
            if match:
                if tempat_clear:
                    # Jika tempat sudah dipisahkan, ambil day dan month dari match
                    tempat = f"{tempat} {match.group(1)}"
                    print(f"Tempat sudah dipisahkan: {tempat}")
                else:
                    tempat = match.group(1)
                day = match.group(2)

        # Validasi hasil
        if all([tempat, day, month, year]) and day.isdigit() and year.isdigit():
            return tempat, day, month, year
        else:
            return None, None, None, None
    except Exception as e:
        print(f"Gagal ekstrak tgl lahir di meet_tgllength: {e}")
        return None, None, None, None
    
def notmeet_tgllength_jadi_satu(data: str) -> tuple:
    """
    Ekstrak tempat, hari, bulan, dan tahun dari string TTL yang menyatu 
    (misal: 'Surabaya12051990' atau 'Medan-12/05/1990').

    Args:
        data (str): String hasil OCR tempat/tanggal lahir tanpa pemisah konsisten.

    Returns:
        tuple: (tempat, day, month_raw, year) atau (None, None, None, None) jika gagal.
    """
    try:
        colon_match = re.search(r"[：:]\s*(\S+)", data)
        if colon_match:
            data = colon_match.group(1)
        match = re.match(r"([A-Za-z]+)([\d\-\/\.]+)", data)
        if match:
            tempat = match.group(1)
            angka = re.findall(r"\d+", match.group(2))
            if len(angka) == 3:
                day, month_raw, year = angka
            elif len(angka) == 1 and len(angka[0]) == 8:
                # Format tanpa pemisah, misal 12051990
                day = angka[0][:2]
                month_raw = angka[0][2:4]
                year = angka[0][4:]
            return tempat, day, month_raw, year
        else:
            # Fallback: ekstrak semua huruf dan angka
            data = correct_alphabets_to_digits(data)
            tempat = ''.join(re.findall(r"[A-Za-z]+", data))
            angka = ''.join(re.findall(r"\d+", data))
            if len(angka) == 8:
                day = angka[:2]
                month_raw = angka[2:4]
                year = angka[4:]
            return tempat, day, month_raw, year
    except Exception as e:
        print(f"Gagal ekstrak data ttl jadi satu : {e}")
        return None, None, None, None
    
def notmeet_tgllength_terpisah(data: str) -> tuple:
    """
    Ekstrak tempat, hari, bulan, dan tahun dari string TTL dengan pemisah 
    (misal: 'Jakarta12-05-1990' atau 'Medan 12/05/1990').

    Args:
        data (str): String hasil OCR tempat/tanggal lahir dengan pemisah tanggal.

    Returns:
        tuple: (tempat, day, month_raw, year) atau (None, None, None, None) jika gagal.
    """
    try:
        match = re.match(r"([A-Za-z]+)(\d{2})[\/\-.](\d{2})[\/\-.](\d{4})", data)
        if not match:
            match = re.match(r"([A-Za-z]+)[\s\.\-]?(\d{2})(\d{2})[\/\-.](\d{4})", data)
        if match:
            tempat = match.group(1)
            day = match.group(2)
            month_raw = match.group(3)
            year = match.group(4)
        else:
            # Fallback: ekstrak semua huruf dan angka
            data = correct_digits_to_alphabets(data)
            tempat = ''.join(re.findall(r"[A-Za-z]+", data))
            angka = ''.join(re.findall(r"\d+", data))
            if len(angka) == 8:
                day = angka[:2]
                month_raw = angka[2:4]
                year = angka[4:]
        return tempat, day, month_raw, year
    except Exception as e:
        print(f"Gagal ekstrak data ttl jadi satu : {e}")
        return None, None, None, None

def notmeet_tgllength(data: str, next_line: bool) -> tuple:
    """
    Mengekstrak tempat, hari, bulan, dan tahun dari string tanggal lahir yang formatnya tidak standar.

    Args:
        data (str): String hasil OCR yang mengandung tempat/tanggal lahir.
        next_line (bool): True jika data berasal dari baris berikutnya.

    Returns:
        tuple: (tempat, day, month, year) jika berhasil, jika gagal semuanya None.
    """
    if not data or len(data) == 0:
        return None, None, None, None

    try:
        tempat, day, month, year = None, None, None, None
        match = None

        if not next_line:
            # Contoh: "Tempat:KOTA12-05-1990"
            tempat, day, month_raw, year = notmeet_tgllength_jadi_satu(data)
        else:
            # Contoh: "KOTA12/05/1990" atau "KOTA 12051990"
            tempat, day, month_raw, year = notmeet_tgllength_terpisah(data)
            
        # Mapping bulan jika perlu
        if month_raw:
            month = bulan_dict.get(month_raw.lower(), month_raw.zfill(2) if month_raw.isdigit() else None)
        else:
            month = None

        if all([tempat, day, month, year]) and day.isdigit() and year.isdigit():
            return tempat, day, month, year
        else:
            return None, None, None, None

    except Exception as e:
        print(f"Gagal ekstrak tgl lahir di notmeet_tgllength: {e}")
        return None, None, None, None

def correct_alphabets_to_digits(input_string):
    # def correct_string(input_string, correction_mapping):
    """
    Corrects characters in the input string using the provided correction mapping.
    A character is corrected only if it is between digits.

    Parameters:
        input_string (str): The string to process.
        correction_mapping (dict): A dictionary mapping characters to their replacements.

    Returns:
        str: The corrected string with replacements applied.
    """

    corrected = []
    n = len(input_string)

    for i, char in enumerate(input_string):
        # Check if the current character is in correction_mapping
        if char in symbol_mapping:
            corrected.append(str(symbol_mapping[char]))

        elif char in alphabet_mapping:
            # Check if the character is between digits
            if (
                i > 0
                and i < n - 1
                and input_string[i - 1].isdigit()
                and input_string[i + 1].isdigit()
            ):
                corrected.append(str(alphabet_mapping[char]))
            else:
                corrected.append(char)  # Skip correction
        else:
            corrected.append(char)  # Append as is if not in correction_mapping

    return "".join(corrected)

def correct_digits_to_alphabets(input_string):
    # def correct_string(input_string, correction_mapping):
    """
    Corrects characters in the input string using the provided correction mapping.
    A character is corrected only if it is between digits.

    Parameters:
        input_string (str): The string to process.
        correction_mapping (dict): A dictionary mapping characters to their replacements.

    Returns:
        str: The corrected string with replacements applied.
    """

    corrected = []
    n = len(input_string)

    for i, char in enumerate(input_string):
        # Check if the current character is in correction_mapping
        if char in digit_mapping:
            # Check if the character is between digits
            if (
                i > 0
                and i < n - 1
                and input_string[i - 1].isalpha()
                and input_string[i + 1].isalpha()
            ):
                corrected.append(str(digit_mapping[char]))
            else:
                corrected.append(char)  # Skip correction
        else:
            corrected.append(char)  # Append as is if not in correction_mapping

    return "".join(corrected)

def searchmapping(field: str, query: str) -> tuple:
    """
    Melakukan pencarian fuzzy pada mapping field tertentu.

    Args:
        field (str): Nama field yang akan dicari ("pekerjaan", "agama", "status_perkawinan", "jenis_kelamin").
        query (str): Query string yang akan dicocokkan.

    Returns:
        tuple: (hasil_mapping, skor_kemiripan) jika ditemukan, jika tidak ditemukan (None, 0).
    """
    mapping_dict = {
        "pekerjaan": mapping_pekerjaan,
        "agama": mapping_agama,
        "status_perkawinan": mapping_status,
        "jenis_kelamin": mapping_jeniskelamin,
    }
    listmap = mapping_dict.get(field)
    if not listmap:
        return None, 0

    match = process.extractOne(query.upper(), listmap, scorer=fuzz.ratio)
    if match:
        return match[0], match[1]
    return None, 0

def clean_colon(text: str) -> str:
    """
    Menghapus karakter ':' dan spasi di awal string.

    Args:
        text (str): Teks yang akan dibersihkan.

    Returns:
        str: Teks tanpa ':' dan spasi di awal.
    """
    if not text:
        return ""
    clean = re.sub(r"^[:\s]+", "", text)
    return clean

def split_tgllahir(text: str) -> list:
    """
    Memecah string tanggal lahir menjadi list bagian-bagian (tempat, hari, bulan, tahun).

    Args:
        text (str): String yang berisi tempat/tanggal lahir.

    Returns:
        list: List hasil split, elemen kosong dihapus.
    """
    if not text:
        return []
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    pattern = r"\b([A-Z][A-Z]+.*)"
    match = re.search(pattern, text)
    text = match.group(1) if match else text
    tgl_split = [part for part in re.split(r"[ /,\-\.]+", text) if part]
    return tgl_split

def getdigitonly(text: str) -> str:
    """
    Mengambil hanya digit dari string (menghapus spasi dan karakter non-digit).

    Args:
        text (str): String yang akan diekstrak digitnya.

    Returns:
        str: String hanya berisi digit.
    """
    if not text:
        return ""
    cleaned = re.sub(r"\D", "", text)
    return cleaned

def matching_nik(data: list) -> str:
    """
    Mengambil dan membersihkan NIK dari data jika confidence mencukupi.

    Args:
        data (list): Data OCR pada baris saat ini, format [text, confidence].

    Returns:
        str: NIK yang sudah dibersihkan, hanya digit. Jika tidak valid, return string kosong.
    """
    nik = data
    if nik and len(nik[0]) > 1 and nik[1] > THRESHOLD_CONFIDENCE:
        clean = clean_colon(nik[0])
        dat = correct_alphabets_to_digits(clean)
        dat = getdigitonly(dat)
        if len(dat) == 16:
            return dat
    return ""

def matching_nama(data: list) -> str:
    """
    Mengambil dan membersihkan nama dari data jika confidence mencukupi.

    Args:
        data (list): Data OCR pada baris saat ini, format [text, confidence].

    Returns:
        str: Nama dalam huruf kapital, atau string kosong jika tidak valid.
    """
    ocr_nama = data
    if ocr_nama and len(ocr_nama[0]) > 1 and ocr_nama[1] > THRESHOLD_CONFIDENCE:
        nama = clean_colon(ocr_nama[0])
        nama = data_verification(nama, "normal")
        # Ganti karakter spesial dengan spasi
        nama = re.sub(r'[^a-zA-Z0-9\s]', ' ', nama)

        # Hilangkan spasi berlebih
        nama = re.sub(r'\s+', ' ', nama).strip()
        return nama.upper()
    return ""

def format_ttl_new_sesuai(data):
    """
    Mengekstrak tempat dan angka tanggal lahir dari data yang sudah sesuai format (misal: ['JAKARTA', '12-05-1990']).

    Args:
        data (list): List berisi dua elemen, [tempat, tanggal_lahir].

    Returns:
        tuple: (tempat, number) berupa string nama tempat dan string angka tanggal lahir (tanpa pemisah).
    """
    try:
        digit_data = len(re.findall(r"\d", str(data[1])))
        if len(data[0]) >= 3 and digit_data >= 5:
            number = re.sub(r'[^a-zA-Z0-9]', '', data[1])
            tempat = data[0]
            print("caps:", tempat)
            print("numbers:", number)
        else:
            # ekstrak tempat
            caps = re.findall(r"[A-Z][A-Z\.\- ]+", data)
            print("caps:", caps)
            # tempat = max(caps, key=len) if caps else ""
            tempat = tempat.replace(".", "").replace("-", " ").strip()

            # ekstrak angka
            numbers = re.findall(r"\d+", data)
            number = "".join(numbers)
            print("numbers:", numbers, "=>", number)
        return tempat, number
    except Exception as e:
        print(f"Gagal format data ttl new sesuai : {e}")
        return "", ""

def format_ttl_new_tidak_sesuai(data):
    """
    Mengekstrak tempat dan angka tanggal lahir dari data yang tidak sesuai format standar.

    Args:
        data (str): String hasil OCR yang mengandung tempat dan tanggal lahir.

    Returns:
        tuple: (tempat, number) berupa string nama tempat dan string angka tanggal lahir (tanpa pemisah).
    """
    try:
        caps = re.findall(r"[A-Z][A-Z\.\- ]+", data)
        print("caps:", caps)
        tempat = max(caps, key=len) if caps else ""
        tempat = tempat.replace(".", "").replace("-", " ").strip()

        # ekstrak angka
        numbers = re.findall(r"\d+", data)
        number = "".join(numbers)
        print("numbers:", numbers, "=>", number)
        return tempat, number
    except Exception as e:
        print(f"Gagal format data ttl new tidak sesuai : {e}")
        return "", ""


def matching_tempatlahir_new(data: list, next_data: list) -> tuple:
    """
    Mengekstrak tempat dan tanggal lahir dari dua baris data OCR dengan format baru.

    Args:
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        tuple: (tempat, tanggal_lahir) jika berhasil, jika gagal ('', '').
    """
    dat = data[0] if data and len(data) > 0 else ""
    next_dat = next_data[0] if next_data and len(next_data) > 0 else ""

    try:
        digit_data = len(re.findall(r"\d", str(dat)))
        digit_nextdata = len(re.findall(r"\d", str(next_dat)))

        # pilih kandidat mana yg lebih cocok
        proses = dat if digit_data >= 6 and digit_nextdata < 6 else next_dat
        if len(proses) <= 12:  # kalau terlalu pendek, gabungkan
            proses = f"{dat} {next_dat}"
        print("proses:", proses)

        # cari teks kapital
        pattern = r"([A-Z]{2,}.*)"
        match = re.search(pattern, proses)
        if not match:
            print("Tidak ada match")
            return "", ""

        content = match.group(1)
        print("match:", content)

        content = re.split(r'[,.]', content)
        if len(content) == 2:
            tempat, number = format_ttl_new_sesuai(content)
        else:
            # ekstrak tempat
            tempat, number = format_ttl_new_tidak_sesuai(proses)

        if number and not number.isdigit():
            number = correct_alphabets_to_digits(number)
            print("corrected numbers:", number)

        day = month_raw = year = None
        if len(number) == 8:  # format DDMMYYYY
            day = number[:2]
            month_raw = number[2:4]
            year = number[4:]

        if all([day, month_raw, year, tempat]):
            month = bulan_dict.get(month_raw, month_raw)
            tgl = f"{day} {month} {year}"
            return tempat, tgl

        return "", ""

    except Exception as e:
        print(f"Gagal ekstrak tgl lahir di matching_tempatlahir_new: {e}")
        return "", ""
        
def validate_ttl_data(data, next_data):
    """
    Memvalidasi dan memecah data tempat/tanggal lahir dari dua baris data OCR.

    Args:
        data (str): String hasil OCR pada baris saat ini.
        next_data (str): String hasil OCR pada baris berikutnya.

    Returns:
        list: List bagian-bagian hasil split (tempat, hari, bulan, tahun), atau list kosong jika gagal.
    """
    try:
        digit_data = len(re.findall(r"\d", data))
        digit_nextdata = len(re.findall(r"\d", next_data))

        # Cek jika next_data valid dan mengandung kata "jenis"
        if next_data and len(next_data) > 1 and digit_data >= 6 and digit_nextdata < 6:
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                print("next_data:", next_data)
                match = re.search(r"[：:]\s*(.*)", data)
                if match:
                    dat = match.group(1)
                parts = split_tgllahir(dat)
        elif digit_nextdata >= 6:
            if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
                clean = clean_colon(next_data)
                parts = split_tgllahir(clean)
        return parts
    except Exception as e:
        print(f"Error Validating ttl_data : {e}")
        return []
    
def matching_tempatlahir(data: list, next_data: list) -> tuple:
    """
    Mengekstrak tempat dan tanggal lahir dari dua baris data.

    Args:
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        tuple: (tempat, tanggal_lahir) jika berhasil, jika gagal (None, None).
    """
    parts = []
    tempat, day, month, year = None, None, None, None

    dat = data[0] if data and len(data) > 0 else ""
    next_dat = next_data[0] if next_data and len(next_data) > 0 else ""

    parts = validate_ttl_data(dat, next_dat)

    print("parts:", parts)
    if len(parts) >= 4 and "jenis" not in next_dat.lower():
        tempat, day, month, year = meet_tgllength(parts)
    else:
        # Step 2: fallback — detect word+digit combo directly
        if next_data and len(next_data) > 1 and "jenis" in next_dat.lower():
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                tempat, day, month, year = notmeet_tgllength(dat, False)
        else:
            if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
                tempat, day, month, year = notmeet_tgllength(next_dat, True)

    if tempat and day and month and year:
        tgl = f"{day.zfill(2)}-{month}-{year.zfill(4)}"
        tgl = re.sub(r"[,.-]", " ", tgl)
        tgl = f"{tgl[:2]} {tgl[2:4]} {tgl[4:]}" if len(tgl) == 8 else tgl
        if len(tempat) > 9:
            match = re.findall(r'[A-Z]{2,}', tempat)
            tempat = " ".join(match)
        return tempat, tgl

    return "", ""
    
##Matching Jenis Kelamin
def matching_jeniskelamin(key: str, data: list, next_data: list) -> str:
    """
    Melakukan mapping jenis kelamin menggunakan fuzzy matching.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Hasil mapping jenis kelamin, atau string kosong jika tidak ditemukan.
    """
    clean = ""
    if key == "jenis kelamin":
        if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
            clean = clean_colon(next_data[0])
    else:
        if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
            clean = clean_colon(data[0])
    if clean:
        jeniskelamin, scorea = searchmapping("jenis_kelamin", clean)
        if scorea >= THRESHOLD_RATIO:
            return jeniskelamin
    return ""

def matching_alamat(next_data: list) -> str:
    """
    Mengambil dan membersihkan alamat dari data berikutnya jika confidence mencukupi.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Alamat dalam huruf kapital, atau string kosong jika tidak valid.
    """
    if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
        clean = clean_colon(next_data[0])
        clean = data_verification(clean, "normal")
        return clean.upper()
    return ""

def rtrw_tidak_sesuai_format(rtrw: str) -> tuple:
    """
    Mengekstrak nilai RT dan RW dari string yang tidak sesuai format standar (misal tanpa tanda '/').
    Fungsi ini mencoba membagi digit pada string menjadi dua bagian (RT dan RW) berdasarkan pola angka,
    kemudian mengembalikan hasilnya dalam format tiga digit (zfill).

    Args:
        rtrw (str): String hasil OCR yang mengandung RT dan RW tanpa format standar.

    Returns:
        tuple: (rt, rw) sebagai string tiga digit, atau ('', '') jika gagal ekstrak.
    """
    final = {}
    index = 0
    before = 0
    try:
        for a in rtrw:
            before = int(a)
            if before == 0 and final != {}:
                index += 1
            if int(a) > 1:
                if final.get(f"{index}"):
                    final[f"{index}"] += a
                else:
                    final[f"{index}"] = a
            rt = final.get("0", "")
            rw = final.get("1", "")
        print("final:", final, "rt:", rt, "rw:", rw)
        if rt.isdigit() and rw.isdigit():
            rt = rt.zfill(3)
            rw = rw.zfill(3)
            return rt, rw
    except Exception as e:
        print(f"gagal ekstrak rtrw : {e}")
        return "", ""

def rtrw_sesuai_format(rtrw: str) -> tuple:
    """
    Mengekstrak nilai RT dan RW dari string yang sudah sesuai format standar (misal '001/002' atau '001002').
    Jika terdapat tanda '/', akan dipisah berdasarkan tanda tersebut, jika tidak akan mengambil 3 digit pertama sebagai RT dan sisanya sebagai RW.
    Hasil dikembalikan dalam format tiga digit (zfill).

    Args:
        rtrw (str): String hasil OCR yang mengandung RT dan RW.

    Returns:
        tuple: (rt, rw) sebagai string tiga digit, atau ('', '') jika gagal ekstrak.
    """
    try:
        if "/" in rtrw:
            rt, rw = rtrw.split("/")
        else:
            rt, rw = rtrw[:3], rtrw[3:]
        if rt.isdigit() and rw.isdigit():
            rt = rt.zfill(3)
            rw = rw.zfill(3)
            return rt, rw
    except Exception as e:
        print(f"gagal ekstrak rtrw : {e}")
        return "", ""

def matching_rtrw(next_data: list) -> tuple:
    """
    Mengekstrak RT dan RW dari data berikutnya jika confidence mencukupi.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        tuple: (rt, rw) dalam format string, atau ('', '') jika tidak valid.
    """
    if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
        dat = re.sub(r'[^0-9/](?=\d)', '', next_data[0])
        if len(dat) >= 6:
            return rtrw_sesuai_format(dat)
        else:
            return rtrw_tidak_sesuai_format(dat)
    return "", ""

def matching_keldesa(next_data: list) -> str:
    """
    Mengambil dan membersihkan kelurahan/desa dari data berikutnya jika confidence mencukupi.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Kelurahan/Desa dalam huruf kapital, atau string kosong jika tidak valid.
    """
    if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
        clean = clean_colon(next_data[0])
        clean = data_verification(clean, "normal")
        return clean.upper()
    return ""

def matching_kecamatan(data: list, next_data: list) -> str:
    """
    Mengambil dan membersihkan kecamatan dari data berikutnya jika confidence mencukupi.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Kecamatan dalam huruf kapital, atau string kosong jika tidak valid.
    """
    if len(data[0]) > 11 and data[1] > THRESHOLD_CONFIDENCE:
        process = data[0]
    else:
        process = next_data[0] if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE else ""
    clean = clean_colon(process)
    clean = data_verification(clean, "normal")
    match = re.findall(r'[A-Z]{2,}', clean)
    if match:
        kecamatan = " ".join(match)
        return kecamatan.upper()
    return ""

def matching_agama(key: str, data: list, next_data: list) -> str:
    """
    Melakukan mapping agama menggunakan fuzzy matching.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Hasil mapping agama, atau string kosong jika tidak ditemukan.
    """
    clean = ""
    if key == "agama":
        if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
            clean = clean_colon(next_data[0])
    else:
        if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
            clean = clean_colon(data[0])
    if clean:
        a, scorea = searchmapping("agama", clean)
        if scorea >= THRESHOLD_RATIO:
            return a
    return ""

def matching_status(key: str, data: list, next_data: list) -> str:
    """
    Melakukan mapping status perkawinan menggunakan fuzzy matching.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Hasil mapping status perkawinan, atau string kosong jika tidak ditemukan.
    """
    prefix_sp = None
    if key == "status perkawinan" and data and len(data) > 0 and len(data[0]) < 20:
        if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
            prefix_sp = re.search(r'[A-Z]{2,}.*', next_data[0])
    else:
        if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
            prefix_sp = re.search(r'[A-Z]{2,}.*', data[0])
    if prefix_sp:
        match_sp = prefix_sp.group()
        a, scorea = searchmapping("status_perkawinan", match_sp)
        if scorea >= THRESHOLD_RATIO:
            return a
    return ""

def data_verification(data, valuetype):
    """
    Melakukan verifikasi data hasil OCR berdasarkan tipe nilai.

    Args:
        data (str): Data hasil OCR yang akan diverifikasi.
        valuetype (str): Tipe verifikasi, "normal" untuk field umum, selain itu untuk field khusus (misal NIK).

    Returns:
        str: Data yang sudah diverifikasi. Jika data termasuk kata kunci yang harus diabaikan, akan mengembalikan string kosong.
    """
    if valuetype == "normal":
        if data.lower() in keydata:
            return ""
        else:
            return data
    else:
        if data.lower() in keyid:
            return ""
        else:
            return data

def create_err_message(is_blurry, is_glare, is_rotated):
    """
    Membuat pesan error gabungan berdasarkan kondisi kualitas gambar.

    Args:
        is_blurry (bool): True jika gambar terdeteksi blur.
        is_glare (bool): True jika gambar terdeteksi glare (pantulan cahaya).
        is_rotated (bool): True jika gambar terdeteksi terbalik/rotasi.

    Returns:
        str: Pesan error yang menggabungkan semua kondisi yang terdeteksi.
    """
    message = []
    if is_blurry:
        message.append(message_blur)
    if is_glare:
        message.append(message_glare)
    if is_rotated:
        message.append(message_rotated)
    if len(message) > 1:
        result = " ".join(message[:-1]) + " dan " + message[-1]
    else:
        result = message[0]

    return result