"""
Unit tests for services.predictor module
"""
from unittest.mock import patch, MagicMock
import numpy as np

import pytest

from src.services.predictor import PredictorService


@pytest.fixture(autouse=True)
def mock_threshold_provider():
    """Provide a stub ThresholdProvider so PredictorService.confidence /
    iou_threshold resolve without a real init_provider() lifespan call."""
    thresholds = {"confidence": 0.5, "iou_threshold": 0.45}
    with patch("src.services.predictor.get_provider") as mock_get_provider:
        provider = MagicMock()
        provider.get.side_effect = lambda key: thresholds[key]
        mock_get_provider.return_value = provider
        yield provider


@pytest.mark.unit
class TestPredictorService:
    """Test PredictorService class"""
    
    def test_init(self):
        """Test PredictorService initialization"""
        predictor = PredictorService()
        
        assert predictor.model is None
        assert predictor.device is None
        assert predictor.device_info is None
        assert predictor.confidence > 0
        assert predictor.iou_threshold > 0
    
    @patch('src.services.predictor.YOLO')
    @patch('src.services.predictor.get_device')  # Patch where it's used, not where defined
    @patch('src.services.predictor.log_device_info')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_model_pytorch(self, mock_exists, mock_log, mock_get_device, mock_yolo, mock_yolo_model):
        """Test loading PyTorch model"""
        mock_get_device.return_value = ('cpu', {'type': 'CPU', 'count': 4})
        mock_yolo.return_value = mock_yolo_model
        
        predictor = PredictorService()
        
        with patch.object(predictor, 'export_format', 'pt'):
            predictor.load_model()
        
        assert predictor.model is not None
        assert predictor.device == 'cpu'
        mock_yolo.assert_called_once()
    
    @patch('src.services.predictor.YOLO')
    @patch('src.services.predictor.get_device')
    @patch('src.services.predictor.log_device_info')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_model_openvino(self, mock_exists, mock_log, mock_get_device, mock_yolo, mock_yolo_model):
        """Test loading OpenVINO model"""
        mock_get_device.return_value = ('cpu', {'type': 'CPU', 'count': 4})
        mock_yolo.return_value = mock_yolo_model
        
        predictor = PredictorService()
        predictor.export_format = 'openvino'
        
        with patch.object(predictor, '_warmup_model'):
            predictor.load_model()
        
        assert predictor.model is not None
    
    @patch('src.services.predictor.get_device', return_value=('cpu', {'type': 'CPU'}))
    @patch('src.services.predictor.log_device_info')
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_model_file_not_found(self, mock_exists, mock_log, mock_get_device):
        """Test loading model when file doesn't exist"""
        predictor = PredictorService()
        
        with pytest.raises(FileNotFoundError):
            predictor.load_model()
    
    @patch('src.services.predictor.YOLO')
    @patch('src.services.predictor.get_device')
    @patch('src.services.predictor.log_device_info')
    @patch('src.services.predictor.config')  # Patch config in predictor module
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_model_gpu(self, mock_exists, mock_config, mock_log, mock_get_device, mock_yolo, mock_yolo_model):
        """Test loading model with GPU"""
        mock_get_device.return_value = ('cuda', {'type': 'GPU', 'name': 'Tesla', 'cuda_available': True})
        mock_yolo.return_value = mock_yolo_model
        
        # Mock config to return PyTorch model path
        mock_config.model_path = './src/models/test_model.pt'
        mock_config.confidence = 0.5
        mock_config.iou_threshold = 0.45
        mock_config.class_names = {0: 'ktp', 1: 'non-ktp'}
        mock_config.get.side_effect = lambda key, default=None: {
            'model.export_format': 'pt',
            'model.openvino_path': None
        }.get(key, default)
        
        predictor = PredictorService()
        predictor.load_model()
        
        # Should call .to() for GPU with PyTorch model
        mock_yolo_model.to.assert_called_with('cuda')
    
    def test_warmup_model(self, mock_yolo_model):
        """Test model warmup"""
        predictor = PredictorService()
        predictor.model = mock_yolo_model
        predictor.device = 'cpu'
        
        # Should not raise exception
        predictor._warmup_model()
        
        # Model should have been called
        assert mock_yolo_model.called
    
    def test_get_openvino_path_exists(self):
        """Test getting OpenVINO model path when it exists"""
        predictor = PredictorService()
        base_path = "/models/test.pt"
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_dir', return_value=True):
                result = predictor._get_openvino_path(base_path)
                assert result is not None
    
    def test_get_openvino_path_not_exists(self):
        """Test getting OpenVINO model path when it doesn't exist"""
        predictor = PredictorService()
        base_path = "/models/test.pt"
        
        with patch('pathlib.Path.exists', return_value=False):
            result = predictor._get_openvino_path(base_path)
            assert result is None
    
    def test_predict(self, mock_yolo_model, sample_image):
        """Test prediction on an image"""
        predictor = PredictorService()
        predictor.model = mock_yolo_model
        predictor.device = 'cpu'
        predictor.class_names = {0: 'ktp', 1: 'non-ktp'}
        
        result = predictor.predict(sample_image, "test.jpg")
        
        # predict returns a list of detections
        assert isinstance(result, list)
    
    def test_predict_with_detections(self, sample_image):
        """Test prediction with actual detections"""
        # Create mock model with detection
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [100.0, 100.0, 200.0, 200.0]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].item.return_value = 0
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].item.return_value = 0.85
        
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        
        predictor = PredictorService()
        predictor.model = mock_model
        predictor.device = 'cpu'
        predictor.class_names = {0: 'ktp', 1: 'non-ktp'}
        
        result = predictor.predict(sample_image, "test.jpg")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['class_name'] == 'ktp'
        assert result[0]['confidence'] == 0.85
    
    def test_predict_no_detections(self, sample_image):
        """Test prediction with no detections"""
        # Create mock model with no detections
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        
        predictor = PredictorService()
        predictor.model = mock_model
        predictor.device = 'cpu'
        predictor.class_names = {0: 'ktp', 1: 'non-ktp'}
        
        result = predictor.predict(sample_image, "test.jpg")
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_is_loaded_true(self, mock_yolo_model):
        """Test is_loaded when model is loaded"""
        predictor = PredictorService()
        predictor.model = mock_yolo_model
        
        assert predictor.is_loaded()
    
    def test_is_loaded_false(self):
        """Test is_loaded when model is not loaded"""
        predictor = PredictorService()
        predictor.model = None
        
        assert not predictor.is_loaded()
    
    def test_get_device_string(self):
        """Test get_device_string"""
        predictor = PredictorService()
        predictor.device = 'cuda:0'
        
        assert predictor.get_device_string() == 'cuda:0'
    
    def test_get_device_info(self):
        """Test get_device_info"""
        predictor = PredictorService()
        predictor.device_info = {'type': 'GPU', 'name': 'Tesla'}
        
        info = predictor.get_device_info()
        assert info['type'] == 'GPU'
        assert info['name'] == 'Tesla'
    
    def test_predict_model_not_loaded(self, sample_image):
        """Test prediction when model is not loaded"""
        predictor = PredictorService()
        predictor.model = None

        with pytest.raises(Exception):
            predictor.predict(sample_image, "test.jpg")

    @pytest.mark.unit
    def test_warmup_model_failure_non_critical(self):
        """Test that warmup failure is non-critical and does not raise"""
        predictor = PredictorService()
        predictor.model = MagicMock(side_effect=Exception("warmup boom"))
        predictor.device = 'cpu'

        # Should NOT raise -- warmup failure is logged as a warning only
        predictor._warmup_model()

    @pytest.mark.unit
    def test_predict_class_name_from_result_names(self, sample_image):
        """Test class name resolution falls back to result.names when not in self.class_names"""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [10.0, 20.0, 30.0, 40.0]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].item.return_value = 2
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].item.return_value = 0.9

        mock_result.boxes = [mock_box]
        mock_result.names = {2: "passport"}
        mock_model.return_value = [mock_result]

        predictor = PredictorService()
        predictor.model = mock_model
        predictor.device = 'cpu'
        predictor.class_names = {}  # empty -- bypass first check

        result = predictor.predict(sample_image, "test.jpg")

        assert len(result) == 1
        assert result[0]['class_name'] == 'passport'

    @pytest.mark.unit
    def test_predict_class_name_fallback_unknown(self, sample_image):
        """Test class name fallback to 'class_{id}' for unknown class ids"""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [10.0, 20.0, 30.0, 40.0]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].item.return_value = 5
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].item.return_value = 0.7

        mock_result.boxes = [mock_box]
        mock_result.names = {}  # class_id 5 not here either
        mock_model.return_value = [mock_result]

        predictor = PredictorService()
        predictor.model = mock_model
        predictor.device = 'cpu'
        predictor.class_names = {}

        result = predictor.predict(sample_image, "test.jpg")

        assert len(result) == 1
        assert result[0]['class_name'] == 'class_5'

    @pytest.mark.unit
    def test_predict_class_name_default_ktp(self, sample_image):
        """Test class name defaults to 'ktp' for class_id 0 when not in any mapping"""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [10.0, 20.0, 30.0, 40.0]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].item.return_value = 0
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].item.return_value = 0.8

        mock_result.boxes = [mock_box]
        mock_result.names = {}  # class_id 0 not here
        mock_model.return_value = [mock_result]

        predictor = PredictorService()
        predictor.model = mock_model
        predictor.device = 'cpu'
        predictor.class_names = {}

        result = predictor.predict(sample_image, "test.jpg")

        assert len(result) == 1
        assert result[0]['class_name'] == 'ktp'

    @pytest.mark.unit
    def test_predict_class_name_default_non_ktp(self, sample_image):
        """Test class name defaults to 'non-ktp' for class_id 1 when not in any mapping"""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [10.0, 20.0, 30.0, 40.0]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].item.return_value = 1
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].item.return_value = 0.75

        mock_result.boxes = [mock_box]
        mock_result.names = {}  # class_id 1 not here
        mock_model.return_value = [mock_result]

        predictor = PredictorService()
        predictor.model = mock_model
        predictor.device = 'cpu'
        predictor.class_names = {}

        result = predictor.predict(sample_image, "test.jpg")

        assert len(result) == 1
        assert result[0]['class_name'] == 'non-ktp'
