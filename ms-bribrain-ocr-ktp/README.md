# ms-bribrain-ppocr

This project uses PaddleOCR to extract text from images and processes the text using various utility functions. The extracted text is then uploaded to Google Cloud Storage.

## Setup

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/ms-bribrain-ppocr.git
    cd ms-bribrain-ppocr
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add the following variables:

    ```env
    GCP_PROJECT_ID=your-gcp-project-id
    SA_CLIENT_EMAIL=your-service-account-email
    SA_PRIVATE_KEY=your-service-account-private-key
    BUCKET_NAME=your-gcs-bucket-name
    ```

## Usage

1. Run the FastAPI server:

    ```sh
    uvicorn main:app --reload
    ```

2. Use an API client like Postman or `curl` to send a POST request to the `/v1/ppocr` endpoint with an image file:

    ```sh
    curl -X POST "http://127.0.0.1:8000/v1/ppocr" -F "file=@path_to_your_image.jpg"
    ```

3. The response will contain the extracted text lines.

## Project Structure

- `main.py`: Contains the FastAPI application and the endpoint for OCR processing.
- `utils.py`: Contains utility functions for processing text and images.
- `requirements.txt`: Lists the dependencies required for the project.
