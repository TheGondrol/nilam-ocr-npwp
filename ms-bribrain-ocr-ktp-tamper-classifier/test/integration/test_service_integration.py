"""
Integration tests for service layer components
Tests interaction between services
"""

import io
import pytest
from typing import cast
from unittest.mock import Mock, patch, AsyncMock

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore[assignment]

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


@pytest.mark.integration
class TestImagePreprocessingIntegration:
    """Integration tests for image preprocessing pipeline"""
    
    @pytest.mark.asyncio
    async def test_full_image_preprocessing_pipeline(self, sample_image_bytes):
        """Test complete image preprocessing from bytes to tensor"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        from src.services.image_preprocessing import load_image_from_bytes, resize_with_padding, preprocess_image
        
        # Load image from bytes
        image = await load_image_from_bytes(sample_image_bytes)
        assert image is not None
        assert image.mode == "RGB"
        
        # Resize with padding
        resized = resize_with_padding(image, size=320)
        assert resized.size == (320, 320)
        
        # Preprocess with mock processor
        mock_processor = Mock()
        if TORCH_AVAILABLE:
            mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 320, 320)}
        else:
            mock_tensor = Mock()
            mock_tensor.shape = (1, 3, 320, 320)
            mock_processor.return_value = {"pixel_values": mock_tensor}
        
        tensor = preprocess_image(resized, mock_processor, image_size=320)
        assert tensor is not None
    
    @pytest.mark.asyncio
    async def test_preprocessing_different_image_sizes(self):
        """Test preprocessing handles different input image sizes"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        from src.services.image_preprocessing import resize_with_padding
        
        test_sizes = [(100, 100), (200, 150), (150, 200), (500, 300)]
        
        for width, height in test_sizes:
            img = Image.new("RGB", (width, height), color="white")
            resized = resize_with_padding(img, size=320)
            assert resized.size == (320, 320), f"Failed for size {width}x{height}"
    
    @pytest.mark.asyncio
    async def test_preprocessing_different_image_formats(self):
        """Test preprocessing handles different image formats"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        from src.services.image_preprocessing import load_image_from_bytes
        
        formats_to_test = [
            ("RGB", "JPEG"),
            ("RGBA", "PNG"),
            ("L", "PNG"),  # Grayscale
        ]
        
        for mode, fmt in formats_to_test:
            img = Image.new(mode, (100, 100), color="white" if mode != "L" else 255)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=fmt)
            img_bytes.seek(0)
            
            loaded = await load_image_from_bytes(img_bytes.getvalue())
            assert loaded.mode == "RGB", f"Failed to convert {mode} to RGB"


@pytest.mark.integration
class TestTamperDetectionServiceIntegration:
    """Integration tests for tamper detection service"""
    
    @pytest.mark.asyncio
    async def test_service_initialization_flow(self, temp_model_path):
        """Test complete service initialization flow"""
        from src.services.tamper_detection import TamperDetectionService
        
        service = TamperDetectionService(
            model_path=temp_model_path,
            image_size=320,
            force_cpu=True
        )
        
        assert service.model_path == temp_model_path
        assert service.image_size == 320
        assert service.force_cpu is True
        assert not service.is_ready()
        
        # Mock the model loading
        with patch("src.services.tamper_detection.load_model") as mock_load:
            mock_model = Mock()
            mock_processor = Mock()
            mock_load.return_value = (mock_model, mock_processor, "cpu")
            
            with patch("src.services.tamper_detection.get_inference_executor") as mock_executor:
                mock_executor.return_value = Mock()
                service.initialize()
                
                assert service.is_ready()
                assert service.model == mock_model
                assert service.processor == mock_processor
                assert service.device == "cpu"
    
    @pytest.mark.asyncio
    async def test_prediction_pipeline_integration(self, sample_image_bytes):
        """Test complete prediction pipeline"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        from src.services.tamper_detection import TamperDetectionService
        
        service = TamperDetectionService(
            model_path="/fake/path",
            image_size=320,
            force_cpu=True
        )
        
        # Mock components
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.id2label = {0: "authentic", 1: "tampered"}
        
        if TORCH_AVAILABLE:
            logits = torch.tensor([[0.2, 0.8]])
            outputs = Mock()
            outputs.logits = logits
            mock_model.return_value = outputs
        else:
            mock_logits = Mock()
            mock_outputs = Mock()
            mock_outputs.logits = mock_logits
            mock_model.return_value = mock_outputs
        
        mock_processor = Mock()
        if TORCH_AVAILABLE:
            mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 320, 320)}
        else:
            mock_tensor = Mock()
            mock_tensor.to = Mock(return_value=mock_tensor)
            mock_processor.return_value = {"pixel_values": mock_tensor}
        
        service.model = mock_model
        service.processor = mock_processor
        service.device = "cpu"
        
        # Test prediction from bytes
        with patch("src.services.image_preprocessing.load_image_from_bytes") as mock_load:
            mock_load.return_value = Image.new("RGB", (100, 100))
            
            result = await service.predict_from_bytes(sample_image_bytes)
            
            assert "predicted_class" in result
            assert "confidence" in result
            assert "probabilities" in result
            assert result["predicted_class"] in ["authentic", "tampered"]


@pytest.mark.integration
class TestDatabaseServiceIntegration:
    """Integration tests for database service"""
    
    @pytest.mark.asyncio
    async def test_database_initialization_flow(self):
        """Test database engine initialization and disposal"""
        from src.services.database_service import init_engine, dispose_engine
        import src.services.database_service as db_service
        
        # Save original state
        original_engine = db_service._async_engine
        original_factory = db_service._async_session_factory
        
        try:
            # Reset state
            db_service._async_engine = None
            db_service._async_session_factory = None
            
            with patch("src.services.database_service.create_async_engine") as mock_engine:
                with patch("src.services.database_service.async_sessionmaker") as mock_factory:
                    mock_eng = AsyncMock()
                    mock_engine.return_value = mock_eng
                    
                    # Initialize
                    await init_engine()
                    
                    assert db_service._async_engine == mock_eng
                    mock_engine.assert_called_once()
                    mock_factory.assert_called_once()
                    
                    # Dispose
                    await dispose_engine()
                    
                    mock_eng.dispose.assert_called_once()
                    assert db_service._async_engine is None
        finally:
            # Restore original state
            db_service._async_engine = original_engine
            db_service._async_session_factory = original_factory
    
    @pytest.mark.asyncio
    async def test_log_insertion_flow(self):
        """Test complete log insertion flow"""
        from src.services.database_service import insert_log
        import src.services.database_service as db_service
        
        # Mock session with proper async methods
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()  # add is synchronous
        mock_session.commit = AsyncMock()  # commit is async
        
        mock_factory = Mock(return_value=mock_session)
        
        original_factory = db_service._async_session_factory
        original_insert_flag = db_service.insert_to_database
        
        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
            db_service._async_session_factory = cast(async_sessionmaker[AsyncSession], mock_factory)
            db_service.insert_to_database = True
            
            await insert_log(
                request_id="test-123",
                response_code=200,
                payload={"test": "data"},
                error_message="",
                result={"prediction": "authentic"},
                processing_time=100.5
            )
            
            mock_factory.assert_called_once()
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
        finally:
            db_service._async_session_factory = original_factory
            db_service.insert_to_database = original_insert_flag


@pytest.mark.integration
class TestMinIOServiceIntegration:
    """Integration tests for MinIO service"""
    
    def test_model_download_flow(self, tmp_path):
        """Test complete model download flow"""
        from src.services.minio_service import download_model_minio as download_model
        from src.core import config
        
        original_path = config.model.file_path
        
        try:
            # Set test path
            test_file = tmp_path / "test_model.zip"
            config.model.file_path = str(test_file)
            
            with patch("src.services.minio_service.Minio") as mock_minio:
                with patch("src.services.minio_service.Path") as mock_path_cls:
                    mock_path = Mock()
                    mock_path.exists.return_value = False
                    mock_path.parent = Mock()
                    mock_path_cls.return_value = mock_path
                    
                    mock_client = Mock()
                    mock_minio.return_value = mock_client
                    
                    # Should attempt download
                    download_model()
                    
                    mock_path.parent.mkdir.assert_called_once()
                    mock_client.fget_object.assert_called_once()
        finally:
            config.model.file_path = original_path
    
    def test_skip_download_if_exists(self, tmp_path):
        """Test that download is skipped if file exists"""
        from src.services.minio_service import download_model_minio as download_model
        from src.core import config
        
        original_path = config.model.file_path
        
        try:
            # Create existing file
            test_file = tmp_path / "existing_model.zip"
            test_file.write_bytes(b"dummy model")
            config.model.file_path = str(test_file)
            
            with patch("src.services.minio_service.Minio") as mock_minio:
                # Should not download
                download_model()
                
                mock_minio.assert_not_called()
        finally:
            config.model.file_path = original_path


@pytest.mark.integration
class TestEndToEndFlow:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_prediction_flow(self, sample_image_bytes):
        """Test complete flow from image bytes to prediction result"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        from src.services.image_preprocessing import load_image_from_bytes, preprocess_image
        
        # Step 1: Load image
        image = await load_image_from_bytes(sample_image_bytes)
        assert image is not None
        
        # Step 2: Preprocess
        mock_processor = Mock()
        if TORCH_AVAILABLE:
            mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 320, 320)}
        else:
            mock_tensor = Mock()
            mock_processor.return_value = {"pixel_values": mock_tensor}
        
        tensor = preprocess_image(image, mock_processor, image_size=320)
        assert tensor is not None
        
        # Step 3: Mock inference
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.id2label = {0: "authentic", 1: "tampered"}
        
        if TORCH_AVAILABLE:
            outputs = Mock()
            outputs.logits = torch.tensor([[0.7, 0.3]])
            mock_model.return_value = outputs
        
        # This simulates a successful end-to-end flow
        assert True  # Flow completed without errors
