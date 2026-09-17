"""
Unit tests for image_preprocessing service
"""

import io
import pytest
import torch
from PIL import Image

from src.services.image_preprocessing import (
    resize_with_padding,
    preprocess_image,
    load_image_from_bytes
)


@pytest.mark.unit
class TestResizeWithPadding:
    """Tests for resize_with_padding function"""
    
    def test_resize_square_image(self):
        """Test resizing a square image"""
        img = Image.new("RGB", (100, 100), color="white")
        resized = resize_with_padding(img, size=320)
        
        assert resized.size == (320, 320)
        assert resized.mode == "RGB"
    
    def test_resize_landscape_image(self):
        """Test resizing a landscape image (wider than tall)"""
        img = Image.new("RGB", (200, 100), color="white")
        resized = resize_with_padding(img, size=320)
        
        assert resized.size == (320, 320)
        # Check that padding was added
        assert resized.mode == "RGB"
    
    def test_resize_portrait_image(self):
        """Test resizing a portrait image (taller than wide)"""
        img = Image.new("RGB", (100, 200), color="white")
        resized = resize_with_padding(img, size=320)
        
        assert resized.size == (320, 320)
        assert resized.mode == "RGB"
    
    def test_resize_with_custom_size(self):
        """Test resizing with custom target size"""
        img = Image.new("RGB", (100, 100), color="white")
        resized = resize_with_padding(img, size=224)
        
        assert resized.size == (224, 224)
    
    def test_padding_color_is_white(self):
        """Test that padding uses white color (255)"""
        img = Image.new("RGB", (100, 200), color="black")
        resized = resize_with_padding(img, size=320)
        
        # Check corners should have white padding
        pixels = resized.load()
        assert pixels is not None, "Failed to load pixel data"
        # Top corners should be white (padding)
        assert pixels[0, 0] == (255, 255, 255)
        assert pixels[319, 0] == (255, 255, 255)


@pytest.mark.unit
class TestPreprocessImage:
    """Tests for preprocess_image function"""
    
    def test_preprocess_returns_tensor(self, sample_image, mock_processor):
        """Test that preprocessing returns a tensor"""
        result = preprocess_image(sample_image, mock_processor, image_size=320)
        
        assert isinstance(result, torch.Tensor)
        mock_processor.assert_called_once()
    
    def test_preprocess_calls_processor_with_correct_args(self, sample_image, mock_processor):
        """Test that processor is called with correct arguments"""
        preprocess_image(sample_image, mock_processor, image_size=320)
        
        call_args = mock_processor.call_args
        assert call_args.kwargs["do_resize"] is False
        assert call_args.kwargs["do_center_crop"] is False
        assert call_args.kwargs["return_tensors"] == "pt"
    
    def test_preprocess_with_custom_image_size(self, mock_processor):
        """Test preprocessing with custom image size"""
        img = Image.new("RGB", (100, 100), color="white")
        preprocess_image(img, mock_processor, image_size=224)
        
        # Check that the image passed to processor is 224x224
        call_args = mock_processor.call_args
        processed_img = call_args.args[0]
        assert processed_img.size == (224, 224)


@pytest.mark.unit
class TestLoadImageFromBytes:
    """Tests for load_image_from_bytes function"""
    
    @pytest.mark.asyncio
    async def test_load_valid_image_bytes(self, sample_image_bytes):
        """Test loading valid image bytes"""
        img = await load_image_from_bytes(sample_image_bytes)
        
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
    
    @pytest.mark.asyncio
    async def test_load_image_converts_to_rgb(self):
        """Test that non-RGB images are converted to RGB"""
        # Create RGBA image
        img_rgba = Image.new("RGBA", (100, 100), color=(255, 255, 255, 128))
        img_bytes = io.BytesIO()
        img_rgba.save(img_bytes, format="PNG")
        
        result = await load_image_from_bytes(img_bytes.getvalue())
        
        assert result.mode == "RGB"
    
    @pytest.mark.asyncio
    async def test_load_invalid_image_bytes_raises_error(self):
        """Test that invalid image bytes raise ValueError"""
        invalid_bytes = b"not an image"
        
        with pytest.raises(ValueError, match="Invalid image data"):
            await load_image_from_bytes(invalid_bytes)
    
    @pytest.mark.asyncio
    async def test_load_empty_bytes_raises_error(self):
        """Test that empty bytes raise ValueError"""
        with pytest.raises(ValueError):
            await load_image_from_bytes(b"")
    
    @pytest.mark.asyncio
    async def test_load_image_maintains_dimensions(self):
        """Test that loaded image maintains original dimensions"""
        img = Image.new("RGB", (200, 150), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        
        result = await load_image_from_bytes(img_bytes.getvalue())
        
        assert result.size == (200, 150)
