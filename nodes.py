import torch
import folder_paths
import comfy.model_management
import os
import sys

# Add current directory to path so we can import network_swinir
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from network_swinir import SwinIR


class SwinIRLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
                "model_type": (
                    ["classicalSR", "lightweightSR", "realSR", "denoising"],
                ),
                "upscale": ("INT", {"default": 4, "min": 1, "max": 8}),
                "window_size": ("INT", {"default": 8, "min": 4, "max": 32}),
                "embed_dim": (
                    "INT",
                    {"default": 180, "min": 48, "max": 512, "step": 12},
                ),
                "depths": ("STRING", {"default": "6, 6, 6, 6, 6, 6"}),
                "num_heads": ("STRING", {"default": "6, 6, 6, 6, 6, 6"}),
                "mlp_ratio": (
                    "FLOAT",
                    {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5},
                ),
                "img_size": (
                    "INT",
                    {"default": 128, "min": 32, "max": 256, "step": 8},
                ),
            }
        }

    RETURN_TYPES = ("SWINIR_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "SwinIR"

    def load_model(
        self,
        model_name,
        model_type,
        upscale,
        window_size,
        embed_dim,
        depths,
        num_heads,
        mlp_ratio,
        img_size,
    ):
        model_path = folder_paths.get_full_path("upscale_models", model_name)

        # Parse depths and num_heads
        try:
            depths_list = [int(x.strip()) for x in depths.split(",")]
            num_heads_list = [int(x.strip()) for x in num_heads.split(",")]
        except ValueError:
            raise ValueError("depths and num_heads must be comma-separated integers")

        # Determine structural parameters based on model_type
        upsampler = ""
        resi_connection = "1conv"

        if model_type == "classicalSR":
            upsampler = "pixelshuffle"
        elif model_type == "lightweightSR":
            upsampler = "pixelshuffledirect"
        elif model_type == "realSR":
            upsampler = "nearest+conv"
            resi_connection = "3conv"
        elif model_type == "denoising":
            upsampler = ""  # No upsample for denoising
            if upscale != 1:
                print(
                    f"Warning: model_type is denoising but upscale is {upscale}. SwinIR denoising usually implies upscale=1."
                )

        # Initialize model
        model = SwinIR(
            upscale=upscale,
            in_chans=3,
            img_size=img_size,
            window_size=window_size,
            img_range=1.0,
            depths=depths_list,
            embed_dim=embed_dim,
            num_heads=num_heads_list,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            resi_connection=resi_connection,
        )

        # Load weights
        load_net = torch.load(model_path, map_location="cpu")

        if "params_ema" in load_net:
            state_dict = load_net["params_ema"]
        elif "params" in load_net:
            state_dict = load_net["params"]
        else:
            state_dict = load_net

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return (model,)


class SwinIRRun:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "swinir_model": ("SWINIR_MODEL",),
                "images": ("IMAGE",),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 64, "max": 4096, "step": 64},
                ),
                "overlap": ("INT", {"default": 32, "min": 0, "max": 128, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "SwinIR"

    def upscale(self, swinir_model, images, tile_size, overlap):
        device = comfy.model_management.get_torch_device()
        swinir_model = swinir_model.to(device)

        output_images = []

        for image in images:
            # image is H, W, C, range 0-1
            # Permute to C, H, W
            img_tensor = image.permute(2, 0, 1).unsqueeze(0).to(device)  # 1, C, H, W

            # Pad if needed to be multiple of window_size (handled by model.check_image_size, but check if check_image_size handles non-window alignment)
            # swinir code handles window partitioning padding in forward_features or check_image_size?
            # SwinIR.check_image_size handles padding for window_size alignment.

            with torch.no_grad():
                # Tiled processing to save memory
                b, c, h, w = img_tensor.shape

                # If image is small enough, process directly
                if h <= tile_size and w <= tile_size:
                    output = swinir_model(img_tensor)
                else:
                    # Simple tiling implementation
                    # Using ComfyUI's model_management or just simple loop
                    # For now, let's just implement a simple tiling strategy or use the one from SwinIR demo?
                    # SwinIR main doesn't have tiling code in class, but demo scripts usually do.
                    # I'll implement a basic tile processing:

                    # (This is a simplified tiling, might have artifacts at edges if not handled carefully with overlap)
                    # For V1, I will try to process directly and if OOM, user should reduce tile_size.
                    # Wait, the tool definition has tile_size. I should use it.

                    output = self.tile_process(
                        img_tensor, swinir_model, tile_size, overlap
                    )

            # Output is 1, C, H, W
            output = output.squeeze(0).permute(1, 2, 0).cpu()  # H, W, C
            output = torch.clamp(output, 0, 1)
            output_images.append(output)

        swinir_model.to("cpu")  # Move back to CPU to save VRAM
        return (torch.stack(output_images),)

    def tile_process(self, img, model, tile_size, overlap):
        b, c, h, w = img.shape
        scale = model.upscale

        # Ensure tile_size does not exceed image dimensions for indexing
        actual_tile_h = min(tile_size, h)
        actual_tile_w = min(tile_size, w)

        # Calculate tiling indices
        if h <= tile_size:
            h_idx_list = [0]
        else:
            h_idx_list = list(range(0, h - tile_size, tile_size - overlap))
            if h_idx_list[-1] + tile_size < h:
                h_idx_list.append(h - tile_size)

        if w <= tile_size:
            w_idx_list = [0]
        else:
            w_idx_list = list(range(0, w - tile_size, tile_size - overlap))
            if w_idx_list[-1] + tile_size < w:
                w_idx_list.append(w - tile_size)

        E = torch.zeros(b, c, h * scale, w * scale).type_as(img)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img[
                    ..., h_idx : h_idx + tile_size, w_idx : w_idx + tile_size
                ]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch)
                W[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch_mask)

        output = E.div_(W)
        return output


NODE_CLASS_MAPPINGS = {"SwinIRLoader": SwinIRLoader, "SwinIRRun": SwinIRRun}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwinIRLoader": "SwinIR Model Loader",
    "SwinIRRun": "SwinIR Upscale/Denoise",
}
