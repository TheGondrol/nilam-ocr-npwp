"""
Unit tests for api.routes module
"""
import io
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from fastapi import UploadFile
from PIL import Image

from src.api.routes import (
    _process_image_sync,
    root,
    health_check,
    predict,
    _check_database,
    _check_model,
    readiness,
)


@pytest.mark.unit
class TestProcessImageSync:
    """Test _process_image_sync function"""
    
    def test_process_image_sync_jpeg(self, sample_image_bytes):
        """Test processing JPEG image"""
        image = _process_image_sync(sample_image_bytes)
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
    
    def test_process_image_sync_png(self):
        """Test processing PNG image"""
        # Create PNG image
        img = Image.new('RGB', (100, 100), color='red')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        
        image = _process_image_sync(png_bytes)
        
        assert isinstance(image, Image.Image)
        assert image.mode == 'RGB'
    
    def test_process_image_sync_invalid_data(self):
        """Test processing invalid image data"""
        invalid_bytes = b"not an image"
        
        with pytest.raises(Exception):
            _process_image_sync(invalid_bytes)


@pytest.mark.unit
@pytest.mark.asyncio
class TestRootEndpoint:
    """Test root endpoint"""
    
    async def test_root(self):
        """Test root endpoint returns correct response"""
        response = await root()
        
        assert response.message == "KTP Detection API"
        assert "health" in response.endpoints
        assert "predict" in response.endpoints
        assert response.version == "1.0.0"


@pytest.mark.unit
@pytest.mark.asyncio
class TestHealthCheckEndpoint:
    """Test health check endpoint"""
    
    async def test_health_check_healthy(self):
        """Test health check when model is loaded and DB is up"""
        with patch('src.api.routes.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = True
            mock_predictor.get_device_string.return_value = "cpu"
            mock_predictor.get_device_info.return_value = {"type": "CPU"}

            with patch('src.api.routes._check_database', return_value={"status": "up"}):
                response = await health_check()

                assert response.status == "healthy"
                assert response.model_loaded
                assert response.device == "cpu"
                assert response.checks is not None
                assert response.checks["model"]["status"] == "up"
                assert response.checks["database"]["status"] == "up"

    async def test_health_check_degraded(self):
        """Test health check when model is loaded but DB is down"""
        with patch('src.api.routes.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = True
            mock_predictor.get_device_string.return_value = "cpu"
            mock_predictor.get_device_info.return_value = {"type": "CPU"}

            with patch('src.api.routes._check_database', return_value={"status": "down", "reason": "not initialised"}):
                response = await health_check()

                assert response.status == "degraded"
                assert response.model_loaded

    async def test_health_check_unhealthy(self):
        """Test health check when model is not loaded"""
        with patch('src.api.routes.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = False
            mock_predictor.get_device_string.return_value = "cpu"
            mock_predictor.get_device_info.return_value = {"type": "CPU"}

            with patch('src.api.routes._check_database', return_value={"status": "up"}):
                response = await health_check()

                assert response.status == "unhealthy"
                assert not response.model_loaded


@pytest.mark.unit
@pytest.mark.asyncio
class TestPredictEndpoint:
    """Test predict endpoint"""
    
    async def test_predict_success(self, sample_image_bytes):
        """Test successful prediction"""
        # Create mock file upload
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.read = AsyncMock(return_value=sample_image_bytes)
        
        # Mock predictor to return list of detections (not KTPDetectionResponse)
        mock_detection = [{
            'class_id': 0,
            'class_name': 'ktp',
            'confidence': 0.85,
            'bbox': {
                'x1': 100,
                'y1': 100,
                'x2': 200,
                'y2': 200
            }
        }]
        
        with patch('src.api.routes.predictor') as mock_predictor:
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                with patch('src.api.routes.request_id_ctx') as mock_ctx:
                    mock_ctx.get.return_value = "test-request-123"
                    mock_predictor.is_loaded.return_value = True
                    mock_predictor.predict.return_value = mock_detection
                    
                    with patch('asyncio.get_running_loop') as mock_loop:
                        mock_loop.return_value.run_in_executor = AsyncMock(
                            side_effect=[
                                Image.open(io.BytesIO(sample_image_bytes)),
                                mock_detection
                            ]
                        )
                        
                        response = await predict(mock_file)

                        import json
                        body = json.loads(response.body.decode())
                        assert body["status_code"] == 200
                        assert body["data"]["filename"] == "test.jpg"
                        assert body["data"]["detected"]
                        assert body["data"]["num_detected"] == 1
    
    async def test_predict_model_not_loaded(self, sample_image_bytes):
        """Test prediction when model is not loaded"""
        import json

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.read = AsyncMock(return_value=sample_image_bytes)

        with patch('src.api.routes.predictor') as mock_predictor:
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                with patch('src.api.routes.request_id_ctx') as mock_ctx:
                    mock_ctx.get.return_value = "test-request-123"
                    mock_predictor.is_loaded.return_value = False

                    response = await predict(mock_file)
                    body = json.loads(response.body.decode())
                    assert response.status_code == 503
                    assert body["error_code"] == "MODEL_NOT_LOADED"
    
    async def test_predict_invalid_file_type(self):
        """Test prediction with invalid file type"""
        import json

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.txt"
        mock_file.content_type = "text/plain"

        with patch('src.api.routes.predictor') as mock_predictor:
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                with patch('src.api.routes.request_id_ctx') as mock_ctx:
                    mock_ctx.get.return_value = "test-request-123"
                    mock_predictor.is_loaded.return_value = True

                    response = await predict(mock_file)
                    body = json.loads(response.body.decode())
                    assert response.status_code == 400
                    assert body["error_code"] == "INVALID_FILE_TYPE"
    
    async def test_predict_image_processing_error(self):
        """Test prediction with image processing error"""
        import json

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.read = AsyncMock(return_value=b"invalid image data")

        with patch('src.api.routes.predictor') as mock_predictor:
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                with patch('src.api.routes.request_id_ctx') as mock_ctx:
                    mock_ctx.get.return_value = "test-request-123"
                    mock_predictor.is_loaded.return_value = True

                    with patch('asyncio.get_running_loop') as mock_loop:
                        mock_loop.return_value.run_in_executor = AsyncMock(
                            side_effect=Exception("Failed to decode image")
                        )

                        response = await predict(mock_file)
                        body = json.loads(response.body.decode())
                        assert response.status_code == 400
                        assert body["error_code"] == "IMAGE_PROCESSING_FAILED"
    
    async def test_predict_prediction_error(self, sample_image_bytes):
        """Test prediction with prediction error"""
        import json

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.read = AsyncMock(return_value=sample_image_bytes)

        with patch('src.api.routes.predictor') as mock_predictor:
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                with patch('src.api.routes.request_id_ctx') as mock_ctx:
                    mock_ctx.get.return_value = "test-request-123"
                    mock_predictor.is_loaded.return_value = True

                    with patch('asyncio.get_running_loop') as mock_loop:
                        mock_loop.return_value.run_in_executor = AsyncMock(
                            side_effect=[
                                Image.open(io.BytesIO(sample_image_bytes)),
                                Exception("Prediction failed")
                            ]
                        )

                        response = await predict(mock_file)
                        body = json.loads(response.body.decode())
                        assert response.status_code == 500
                        assert body["error_code"] == "PREDICTION_FAILED"


@pytest.mark.unit
@pytest.mark.asyncio
class TestCheckDatabase:
    """Test _check_database helper"""

    async def test_check_database_session_factory_none(self):
        """Test _check_database when _async_session_factory is None"""
        with patch(
            'src.services.database_service._async_session_factory',
            None,
        ):
            result = await _check_database()
            assert result["status"] == "down"
            assert result["reason"] == "not initialised"

    async def test_check_database_execute_raises(self):
        """Test _check_database when session.execute raises an Exception"""
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("connection refused")

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_db_module = MagicMock(_async_session_factory=mock_factory)

        with patch.dict('sys.modules', {'src.services.database_service': mock_db_module}):
            result = await _check_database()
            assert result["status"] == "down"
            assert "connection refused" in result["reason"]


@pytest.mark.unit
class TestCheckModel:
    """Test _check_model helper"""

    def test_check_model_exception(self):
        """Test _check_model when predictor.is_loaded() raises an Exception"""
        with patch('src.api.routes.predictor') as mock_predictor:
            mock_predictor.is_loaded.side_effect = RuntimeError("corrupted state")

            result = _check_model()
            assert result["status"] == "down"
            assert "corrupted state" in result["reason"]


@pytest.mark.unit
@pytest.mark.asyncio
class TestReadinessEndpoint:
    """Test readiness probe endpoint"""

    async def test_readiness_both_up(self):
        """Test readiness returns 200 when model and database are both up"""
        with patch('src.api.routes._check_model', return_value={"status": "up", "device": "cpu"}):
            with patch(
                'src.api.routes._check_database',
                new_callable=AsyncMock,
                return_value={"status": "up"},
            ):
                response = await readiness()
                assert response.status_code == 200
                import json
                body = json.loads(response.body.decode())
                assert body["status"] == "ready"

    async def test_readiness_model_down(self):
        """Test readiness returns 503 when model is down"""
        with patch(
            'src.api.routes._check_model',
            return_value={"status": "down", "reason": "model not loaded"},
        ):
            with patch(
                'src.api.routes._check_database',
                new_callable=AsyncMock,
                return_value={"status": "up"},
            ):
                response = await readiness()
                assert response.status_code == 503
                import json
                body = json.loads(response.body.decode())
                assert body["status"] == "not_ready"

    async def test_readiness_db_down(self):
        """Test readiness returns 503 when database is down"""
        with patch('src.api.routes._check_model', return_value={"status": "up", "device": "cpu"}):
            with patch(
                'src.api.routes._check_database',
                new_callable=AsyncMock,
                return_value={"status": "down", "reason": "not initialised"},
            ):
                response = await readiness()
                assert response.status_code == 503
                import json
                body = json.loads(response.body.decode())
                assert body["status"] == "not_ready"


@pytest.mark.unit
@pytest.mark.asyncio
class TestPredictDecisionLogic:
    """Test predict endpoint decision-logic branches"""

    async def _call_predict_with_detections(self, sample_image_bytes, detections):
        """Helper: call predict() with mocked detections and return parsed body."""
        import json

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.read = AsyncMock(return_value=sample_image_bytes)

        with patch('src.api.routes.predictor') as mock_predictor:
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                with patch('src.api.routes.request_id_ctx') as mock_ctx:
                    mock_ctx.get.return_value = "test-request-123"
                    mock_predictor.is_loaded.return_value = True
                    mock_predictor.predict.return_value = detections

                    with patch('asyncio.get_running_loop') as mock_loop:
                        mock_loop.return_value.run_in_executor = AsyncMock(
                            side_effect=[
                                Image.open(io.BytesIO(sample_image_bytes)),
                                detections,
                            ]
                        )

                        response = await predict(mock_file)
                        body = json.loads(response.body.decode())
                        return body

    async def test_predict_only_non_ktp(self, sample_image_bytes):
        """Test predict when only non-KTP objects are detected"""
        detections = [
            {
                'class_id': 1,
                'class_name': 'non-ktp',
                'confidence': 0.90,
                'bbox': {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150},
            }
        ]

        body = await self._call_predict_with_detections(sample_image_bytes, detections)

        assert body["status_code"] == 200
        assert body["data"]["detected"] is False
        assert body["data"]["status"] == "only non-KTP detected"
        assert body["data"]["num_detected"] == 0

    async def test_predict_multiple_ktps(self, sample_image_bytes):
        """Test predict when multiple KTPs are detected (2 ktp, 0 non-ktp)"""
        detections = [
            {
                'class_id': 0,
                'class_name': 'ktp',
                'confidence': 0.85,
                'bbox': {'x1': 10, 'y1': 10, 'x2': 100, 'y2': 100},
            },
            {
                'class_id': 0,
                'class_name': 'ktp',
                'confidence': 0.80,
                'bbox': {'x1': 200, 'y1': 200, 'x2': 300, 'y2': 300},
            },
        ]

        body = await self._call_predict_with_detections(sample_image_bytes, detections)

        assert body["status_code"] == 200
        assert body["data"]["detected"] is True
        assert body["data"]["status"] == "multiple KTPs detected"
        assert body["data"]["num_detected"] == 2

    async def test_predict_mixed_ktp_and_non_ktp(self, sample_image_bytes):
        """Test predict when both KTP and non-KTP are detected (1 ktp + 1 non-ktp)"""
        detections = [
            {
                'class_id': 0,
                'class_name': 'ktp',
                'confidence': 0.85,
                'bbox': {'x1': 10, 'y1': 10, 'x2': 100, 'y2': 100},
            },
            {
                'class_id': 1,
                'class_name': 'non-ktp',
                'confidence': 0.75,
                'bbox': {'x1': 200, 'y1': 200, 'x2': 300, 'y2': 300},
            },
        ]

        body = await self._call_predict_with_detections(sample_image_bytes, detections)

        assert body["status_code"] == 200
        assert body["data"]["detected"] is True
        assert body["data"]["status"] == "KTP and non-KTP detected"
        assert body["data"]["num_detected"] == 1

    async def test_predict_single_ktp_ok(self, sample_image_bytes):
        """Test predict when a single KTP is detected (1 ktp, 0 non-ktp) returns OK"""
        detections = [
            {
                'class_id': 0,
                'class_name': 'ktp',
                'confidence': 0.92,
                'bbox': {'x1': 10, 'y1': 10, 'x2': 200, 'y2': 200},
            },
        ]

        body = await self._call_predict_with_detections(sample_image_bytes, detections)

        assert body["status_code"] == 200
        assert body["data"]["detected"] is True
        assert body["data"]["status"] == "OK"
        assert body["data"]["num_detected"] == 1
        assert body["data"]["reason"] is None
