"""
Test script for SwinIR ComfyUI nodes.
This test creates a mock model and tests the node functionality without requiring a full ComfyUI installation.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Mock folder_paths and comfy.model_management
class MockFolderPaths:
    @staticmethod
    def get_filename_list(category):
        return ["test_model.pth"]
    
    @staticmethod
    def get_full_path(category, filename):
        return os.path.join(current_dir, "test_model.pth")

class MockModelManagement:
    @staticmethod
    def get_torch_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mock the modules
sys.modules['folder_paths'] = MockFolderPaths
sys.modules['comfy'] = type('obj', (object,), {'model_management': MockModelManagement})
sys.modules['comfy.model_management'] = MockModelManagement

from network_swinir import SwinIR
from nodes import SwinIRLoader, SwinIRRun


def create_test_model(save_path="test_model.pth"):
    """Create a minimal SwinIR model for testing."""
    print("Creating test model...")
    model = SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.,
        depths=[2, 2],  # Smaller for testing
        embed_dim=60,
        num_heads=[6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Test model saved to {save_path}")
    return model


def test_loader():
    """Test SwinIRLoader node."""
    print("\n" + "="*50)
    print("Testing SwinIRLoader...")
    print("="*50)
    
    # Create test model if it doesn't exist
    model_path = os.path.join(current_dir, "test_model.pth")
    if not os.path.exists(model_path):
        create_test_model(model_path)
    
    loader = SwinIRLoader()
    
    # Test INPUT_TYPES
    input_types = loader.INPUT_TYPES()
    print(f"✓ INPUT_TYPES: {list(input_types['required'].keys())}")
    
    # Test load_model
    try:
        result = loader.load_model(
            model_name="test_model.pth",
            model_type="classicalSR",
            upscale=2,
            window_size=8,
            embed_dim=60,
            depths="2, 2",
            num_heads="6, 6",
            mlp_ratio=2.0
        )
        model = result[0]
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Model upscale: {model.upscale}")
        print(f"  Model window_size: {model.window_size}")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise


def test_runner(model):
    """Test SwinIRRun node."""
    print("\n" + "="*50)
    print("Testing SwinIRRun...")
    print("="*50)
    
    runner = SwinIRRun()
    
    # Test INPUT_TYPES
    input_types = runner.INPUT_TYPES()
    print(f"✓ INPUT_TYPES: {list(input_types['required'].keys())}")
    
    # Create test image (ComfyUI format: batch of H, W, C in range [0, 1])
    test_image = torch.rand(1, 64, 64, 3)  # 1 image, 64x64, RGB
    print(f"✓ Created test image: {test_image.shape}")
    
    # Test upscale
    try:
        result = runner.upscale(
            swinir_model=model,
            images=test_image,
            tile_size=512,
            overlap=32
        )
        output = result[0]
        print(f"✓ Upscale completed successfully")
        print(f"  Input shape: {test_image.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: (1, 128, 128, 3)")
        
        # Verify output shape
        expected_h = test_image.shape[1] * model.upscale
        expected_w = test_image.shape[2] * model.upscale
        assert output.shape == (1, expected_h, expected_w, 3), \
            f"Output shape mismatch: {output.shape} vs expected (1, {expected_h}, {expected_w}, 3)"
        print(f"✓ Output shape verified")
        
        # Verify output range
        assert output.min() >= 0 and output.max() <= 1, \
            f"Output range invalid: [{output.min()}, {output.max()}]"
        print(f"✓ Output range verified: [{output.min():.4f}, {output.max():.4f}]")
        
    except Exception as e:
        print(f"✗ Failed to upscale: {e}")
        raise


def test_tiling(model):
    """Test tiling functionality with larger image."""
    print("\n" + "="*50)
    print("Testing Tiling...")
    print("="*50)
    
    runner = SwinIRRun()
    
    # Create larger test image that requires tiling
    test_image = torch.rand(1, 600, 800, 3)  # 1 image, 600x800, RGB
    print(f"✓ Created large test image: {test_image.shape}")
    
    try:
        result = runner.upscale(
            swinir_model=model,
            images=test_image,
            tile_size=256,  # Smaller tile to force tiling
            overlap=32
        )
        output = result[0]
        print(f"✓ Tiled upscale completed successfully")
        print(f"  Input shape: {test_image.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Verify output shape
        expected_h = test_image.shape[1] * model.upscale
        expected_w = test_image.shape[2] * model.upscale
        assert output.shape == (1, expected_h, expected_w, 3), \
            f"Output shape mismatch: {output.shape} vs expected (1, {expected_h}, {expected_w}, 3)"
        print(f"✓ Tiled output shape verified")
        
    except Exception as e:
        print(f"✗ Failed tiled upscale: {e}")
        raise


def test_batch_processing(model):
    """Test batch processing."""
    print("\n" + "="*50)
    print("Testing Batch Processing...")
    print("="*50)
    
    runner = SwinIRRun()
    
    # Create batch of test images
    batch_size = 3
    test_images = torch.rand(batch_size, 64, 64, 3)
    print(f"✓ Created batch of {batch_size} test images: {test_images.shape}")
    
    try:
        result = runner.upscale(
            swinir_model=model,
            images=test_images,
            tile_size=512,
            overlap=32
        )
        output = result[0]
        print(f"✓ Batch upscale completed successfully")
        print(f"  Input shape: {test_images.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Verify output shape
        expected_h = test_images.shape[1] * model.upscale
        expected_w = test_images.shape[2] * model.upscale
        assert output.shape == (batch_size, expected_h, expected_w, 3), \
            f"Output shape mismatch: {output.shape} vs expected ({batch_size}, {expected_h}, {expected_w}, 3)"
        print(f"✓ Batch output shape verified")
        
    except Exception as e:
        print(f"✗ Failed batch upscale: {e}")
        raise


def cleanup():
    """Clean up test files."""
    test_model_path = os.path.join(current_dir, "test_model.pth")
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
        print(f"\n✓ Cleaned up test model: {test_model_path}")


def main():
    """Run all tests."""
    print("\n" + "="*50)
    print("SwinIR Node Test Suite")
    print("="*50)
    
    try:
        # Test loader
        model = test_loader()
        
        # Test runner
        test_runner(model)
        
        # Test tiling
        test_tiling(model)
        
        # Test batch processing
        test_batch_processing(model)
        
        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50)
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"✗ Tests failed with error: {e}")
        print("="*50)
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())
