import time
import os
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from utils.utils import mappingnext
from utils.helper import perform_ocr, check_confidencemedian, image_quality, create_err_message

app = FastAPI()

@app.post("/v1/ppocr")
async def extract_text_lines(
    file: UploadFile = File(...)
):
    """
    Endpoint untuk ekstraksi data KTP dari gambar menggunakan OCR.

    Args:
        file (UploadFile): File gambar yang diupload (JPEG/PNG).

    Returns:
        JSONResponse: 
            - 200: Jika berhasil, mengembalikan hasil ekstraksi field KTP.
            - 400: Jika file tidak valid, confidence rendah, atau kualitas gambar buruk.
            - 500: Jika terjadi error internal saat proses.
    """
    try:
        start = time.time()
        # try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid file type. Only JPEG and PNG are supported."},
            )

        # Read image bytes
        image_bytes = await file.read()

        # Perform OCR on the image
        result = perform_ocr(image_bytes)
        is_confidence = check_confidencemedian(result)
        if result is None or is_confidence:
            return JSONResponse(
                status_code=400,
                content={"error": "KTP tidak dapat terdeteksi. Pastikan gambar memiliki kualitas yang baik dan KTP terlihat jelas."},
            )
        
        # Call Detect image quality function
        is_blurry, is_glare, is_rotated = image_quality(image_bytes, result)

        if is_blurry or is_glare or is_rotated:
            print(is_blurry, is_glare, is_rotated, is_confidence)
            message = create_err_message(is_blurry, is_glare, is_rotated)
            return JSONResponse(
                status_code=400,
                content={"error": f"Gambar yang diambil terdeteksi {message}"},
            )
        traversal = mappingnext(result, image_bytes)

        return JSONResponse(status_code=200, content={"text_lines": traversal})

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "An error occurred while processing the image."},
        )