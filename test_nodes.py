"""
Test script for SwinIR ComfyUI nodes.
This test creates a mock model and tests the node functionality without requiring a full ComfyUI installation.
"""

import torch
import numpy as np
import sys
import os

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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
sys.modules["folder_paths"] = MockFolderPaths
sys.modules["comfy"] = type("obj", (object,), {"model_management": MockModelManagement})
sys.modules["comfy.model_management"] = MockModelManagement

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
        img_range=1.0,
        depths=[2, 2],  # Smaller for testing
        embed_dim=60,
        num_heads=[6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Test model saved to {save_path}")
    return model


def test_loader():
    """Test SwinIRLoader node."""
    print("\n" + "=" * 50)
    print("Testing SwinIRLoader...")
    print("=" * 50)

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
            mlp_ratio=2.0,
            img_size=64,
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
    print("\n" + "=" * 50)
    print("Testing SwinIRRun...")
    print("=" * 50)

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
            swinir_model=model, images=test_image, tile_size=512, overlap=32
        )
        output = result[0]
        print(f"✓ Upscale completed successfully")
        print(f"  Input shape: {test_image.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: (1, 128, 128, 3)")

        # Verify output shape
        expected_h = test_image.shape[1] * model.upscale
        expected_w = test_image.shape[2] * model.upscale
        assert output.shape == (1, expected_h, expected_w, 3), (
            f"Output shape mismatch: {output.shape} vs expected (1, {expected_h}, {expected_w}, 3)"
        )
        print(f"✓ Output shape verified")

        # Verify output range
        assert output.min() >= 0 and output.max() <= 1, (
            f"Output range invalid: [{output.min()}, {output.max()}]"
        )
        print(f"✓ Output range verified: [{output.min():.4f}, {output.max():.4f}]")

    except Exception as e:
        print(f"✗ Failed to upscale: {e}")
        raise


def test_tiling(model):
    """Test tiling functionality with larger image."""
    print("\n" + "=" * 50)
    print("Testing Tiling...")
    print("=" * 50)

    runner = SwinIRRun()

    # Create larger test image that requires tiling
    test_image = torch.rand(1, 600, 800, 3)  # 1 image, 600x800, RGB
    print(f"✓ Created large test image: {test_image.shape}")

    try:
        result = runner.upscale(
            swinir_model=model,
            images=test_image,
            tile_size=256,  # Smaller tile to force tiling
            overlap=32,
        )
        output = result[0]
        print(f"✓ Tiled upscale completed successfully")
        print(f"  Input shape: {test_image.shape}")
        print(f"  Output shape: {output.shape}")

        # Verify output shape
        expected_h = test_image.shape[1] * model.upscale
        expected_w = test_image.shape[2] * model.upscale
        assert output.shape == (1, expected_h, expected_w, 3), (
            f"Output shape mismatch: {output.shape} vs expected (1, {expected_h}, {expected_w}, 3)"
        )
        print(f"✓ Tiled output shape verified")

    except Exception as e:
        print(f"✗ Failed tiled upscale: {e}")
        raise


def test_batch_processing(model):
    """Test batch processing."""
    print("\n" + "=" * 50)
    print("Testing Batch Processing...")
    print("=" * 50)

    runner = SwinIRRun()

    # Create batch of test images
    batch_size = 3
    test_images = torch.rand(batch_size, 64, 64, 3)
    print(f"✓ Created batch of {batch_size} test images: {test_images.shape}")

    try:
        result = runner.upscale(
            swinir_model=model, images=test_images, tile_size=512, overlap=32
        )
        output = result[0]
        print(f"✓ Batch upscale completed successfully")
        print(f"  Input shape: {test_images.shape}")
        print(f"  Output shape: {output.shape}")

        # Verify output shape
        expected_h = test_images.shape[1] * model.upscale
        expected_w = test_images.shape[2] * model.upscale
        assert output.shape == (batch_size, expected_h, expected_w, 3), (
            f"Output shape mismatch: {output.shape} vs expected ({batch_size}, {expected_h}, {expected_w}, 3)"
        )
        print(f"✓ Batch output shape verified")

    except Exception as e:
        print(f"✗ Failed batch upscale: {e}")
        raise


def test_attention_mask_mismatch():
    """Test loading model with different parameters (attention mask mismatch scenario)."""
    print("\n" + "=" * 50)
    print("Testing Attention Mask Mismatch Fix...")
    print("=" * 50)

    # Create a model with img_size=128 (like the real denoising checkpoint)
    print("Step 1: Creating model with img_size=128 (creates larger attn_mask)...")
    model_original = SwinIR(
        upscale=1,
        in_chans=3,
        img_size=128,  # This creates attn_mask with shape [256, 64, 64]
        window_size=8,
        img_range=1.0,
        depths=[2, 2],  # Same architecture
        embed_dim=60,  # Same architecture
        num_heads=[6, 6],
        mlp_ratio=2,
        upsampler="",  # Denoising
        resi_connection="1conv",
    )

    # Save the model
    mismatch_model_path = os.path.join(current_dir, "test_mismatch_model.pth")
    torch.save(model_original.state_dict(), mismatch_model_path)
    print(f"✓ Model saved to {mismatch_model_path}")

    # Try to load with DIFFERENT img_size (this is the real error scenario)
    print(
        "\nStep 2: Loading model with img_size=64 (different from checkpoint's 128)..."
    )
    print("  This simulates the original error:")
    print("    Checkpoint attn_mask: [256, 64, 64] (from img_size=128)")
    print("    Model attn_mask:      [64, 64, 64]  (from img_size=64)")

    loader = SwinIRLoader()

    # Update the mock to return the mismatch model
    class MockFolderPathsMismatch:
        @staticmethod
        def get_filename_list(category):
            return ["test_mismatch_model.pth"]

        @staticmethod
        def get_full_path(category, filename):
            return mismatch_model_path

    # Temporarily replace the mock
    original_mock = sys.modules["folder_paths"]
    sys.modules["folder_paths"] = MockFolderPathsMismatch

    try:
        # This should succeed with strict=False, but would fail with strict=True
        result = loader.load_model(
            model_name="test_mismatch_model.pth",
            model_type="denoising",
            upscale=1,
            window_size=8,
            embed_dim=60,  # Same as saved model
            depths="2, 2",  # Same as saved model
            num_heads="6, 6",
            mlp_ratio=2.0,
            img_size=64,  # DIFFERENT from saved (128) - causes attn_mask mismatch
        )
        model_loaded = result[0]
        print("\n✓ Model loaded successfully despite img_size mismatch!")
        print("  Saved with img_size=128 -> attn_mask [256, 64, 64]")
        print("  Loaded with img_size=64 -> attn_mask [64, 64, 64]")
        print("  The strict=False parameter allows this to be skipped")

        # Verify the loaded model has the correct parameters
        assert model_loaded.embed_dim == 60, "Model should use embed_dim=60"
        assert len(model_loaded.layers) == 2, "Model should have 2 layers"
        print("✓ Model architecture parameters verified")

        # Test that the model can actually run inference
        test_img = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            output = model_loaded(test_img)
        # Denoising: upscale=1, output same size as input
        assert output.shape == test_img.shape, (
            f"Output size mismatch: expected {test_img.shape}, got {output.shape}"
        )
        print("✓ Model inference works correctly")
        print(f"  Input: {test_img.shape} -> Output: {output.shape}")

    except Exception as e:
        print(f"✗ Failed to load model with mismatched parameters: {e}")
        raise
    finally:
        # Restore the original mock
        sys.modules["folder_paths"] = original_mock
        # Clean up test file
        if os.path.exists(mismatch_model_path):
            os.remove(mismatch_model_path)
            print(f"\n✓ Cleaned up: {mismatch_model_path}")

    print("\n" + "=" * 50)
    print("✓ Attention mask mismatch test PASSED!")
    print("  The fix (strict=False) successfully handles parameter mismatches")
    print("=" * 50)


def test_real_model_loading():
    """Test loading real pre-trained model from official SwinIR repository."""
    print("\n" + "=" * 50)
    print("Testing Real Model Loading...")
    print("=" * 50)

    # Model details from official SwinIR repository
    model_name = "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
    model_url = (
        f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{model_name}"
    )
    model_path = os.path.join(current_dir, model_name)

    # Download model if not exists
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} from GitHub releases...")
        try:
            r = requests.get(model_url, allow_redirects=True, timeout=60)
            r.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(r.content)
            print(f"✓ Downloaded {model_name} ({len(r.content) / 1024 / 1024:.1f} MB)")
        except requests.RequestException as e:
            print(f"⚠ Could not download model: {e}")
            print("  Skipping real model test (requires internet)")
            return
    else:
        print(f"✓ Using cached model: {model_path}")

    # Official parameters for 005_colorDN model (from main_test_swinir.py)
    print("\nLoading with official 005_colorDN parameters:")
    print("  img_size=128, window_size=8, embed_dim=180")
    print("  depths=[6,6,6,6,6,6], upscale=1, upsampler=''")

    try:
        # Create model with official parameters
        model = SwinIR(
            upscale=1,
            in_chans=3,
            img_size=128,  # Key: must match checkpoint's attn_mask size
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="",
            resi_connection="1conv",
        )

        # Load checkpoint
        load_net = torch.load(model_path, map_location="cpu")
        if "params" in load_net:
            state_dict = load_net["params"]
        elif "params_ema" in load_net:
            state_dict = load_net["params_ema"]
        else:
            state_dict = load_net

        # Load with strict=False (the fix!)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        print("\n✓ Real model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Model embed_dim: {model.embed_dim}")
        print(f"  Model window_size: {model.window_size}")
        print(f"  Model upscale: {model.upscale}")

        # Test inference
        print("\nTesting inference with real model...")
        test_img = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            output = model(test_img)

        # Denoising (upscale=1): output = input size
        assert output.shape == test_img.shape, (
            f"Output shape mismatch: {output.shape} vs {test_img.shape}"
        )
        print("✓ Inference completed successfully")
        print(f"  Input: {test_img.shape} -> Output: {output.shape}")

    except Exception as e:
        print(f"\n✗ Failed to load/run real model: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("\n" + "=" * 50)
    print("✓ Real model loading test PASSED!")
    print("  This validates that strict=False + correct img_size fixes the error")
    print("=" * 50)


def cleanup():
    """Clean up test files."""
    test_model_path = os.path.join(current_dir, "test_model.pth")
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
        print(f"\n✓ Cleaned up test model: {test_model_path}")


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("SwinIR Node Test Suite")
    print("=" * 50)

    try:
        # Test loader
        model = test_loader()

        # Test runner
        test_runner(model)

        # Test tiling
        test_tiling(model)

        # Test batch processing
        test_batch_processing(model)

        # Test with real pre-trained model (the actual fix validation)
        test_real_model_loading()

        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"✗ Tests failed with error: {e}")
        print("=" * 50)
        import traceback

        traceback.print_exc()
        return 1

    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
