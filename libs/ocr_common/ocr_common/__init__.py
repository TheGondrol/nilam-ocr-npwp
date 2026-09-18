"""
Kode bersama untuk keempat service OCR NPWP.

Yang ada di sini adalah hal yang harus SAMA di semua service supaya
orkestrator melihat satu bahasa: envelope response, auth X-API-Key,
request_id, intake file/file_url, klien ke model remote, dan pabrik
FastAPI app. Logika bisnis tiap service tetap di services/<nama>/src.
"""
