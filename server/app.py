"""
AI-сервер для сегментации + перекраски с SAM-2 и HSV blend
Простая перекраска с сохранением текстуры вместо ControlNet Inpaint.
"""
import logging
import time
import traceback
import numpy as np
import torch
from io import BytesIO
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

torch.set_float32_matmul_precision('high')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {_device}")

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
        _predictor.model = _predictor.model.to(_device).eval()
        logger.info("SAM-2 Hiera-L loaded")
    except Exception as e:
        logger.error(f"SAM-2 load error: {e}")
        _predictor = None

    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")


app = FastAPI(title="AI Colorization API", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None
_device = "cpu"


def get_color_hex_name(hex_color: int) -> str:
    from colorsys import rgb_to_hsv
    r = (hex_color >> 16) & 0xFF
    g = (hex_color >> 8) & 0xFF
    b = hex_color & 0xFF
    h, s, v = rgb_to_hsv(r/255, g/255, b/255)
    if s < 0.15:
        if v > 0.8:
            return "white"
        if v < 0.2:
            return "black"
        return "lightgray"
    if h < 0.1:
        return "red"
    if h < 0.2:
        return "yellow"
    if h < 0.4:
        return "green"
    if h < 0.6:
        return "blue"
    if h < 0.8:
        return "purple"
    return "red"


def recolor_image_preserve_texture(image_pil, mask_pil, target_r, target_g, target_b):
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    mask_np = np.array(mask_pil.convert('L')).astype(np.float32) / 255.0
    h_img, w_img = image_pil.size
    result = image_np.copy()

    t_r, t_g, t_b = target_r / 255.0, target_g / 255.0, target_b / 255.0
    t_max = max(t_r, t_g, t_b)
    t_min = min(t_r, t_g, t_b)
    t_diff = t_max - t_min
    if t_max == t_r:
        t_hue = 60 * (((t_g - t_b) / t_diff) % 6) if t_diff > 0 else 0
    elif t_max == t_g:
        t_hue = 60 * ((t_b - t_r) / t_diff + 2) if t_diff > 0 else 120
    else:
        t_hue = 60 * ((t_r - t_g) / t_diff + 4) if t_diff > 0 else 240
    if t_hue < 0:
        t_hue += 360
    t_sat = t_diff / t_max if t_max > 0 else 0

    for y in range(h_img):
        for x in range(w_img):
            idx = y * w_img + x
            if mask_np[idx] > 0.5:
                r, g, b = image_np[y, x]
                value = max(r, g, b)
                c = value * t_sat
                x_val = c * (1 - abs((t_hue / 60) % 2 - 1))
                m = value - c
                if t_hue < 60:
                    nr, ng, nb = c, x_val, 0
                elif t_hue < 120:
                    nr, ng, nb = x_val, c, 0
                elif t_hue < 180:
                    nr, ng, nb = 0, c, x_val
                elif t_hue < 240:
                    nr, ng, nb = 0, x_val, c
                elif t_hue < 300:
                    nr, ng, nb = x_val, 0, c
                else:
                    nr, ng, nb = c, 0, x_val
                result[y, x] = [(nr + m) * 255, (ng + m) * 255, (nb + m) * 255]

    return Image.fromarray(result.astype(np.uint8), 'RGB')


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": _device,
        "models_loaded": _predictor is not None
    }


@app.post("/ai-recolor")
async def ai_recolor(
    image: UploadFile = File(...),
    point_x: float = Form(...),
    point_y: float = Form(...),
    material: str = Form("wood"),
    color_hex: str = Form("0xFF8B4513"),
    strength: float = Form(1.0),
):
    start_time = time.time()
    logger.info("📥 ===== NEW REQUEST =====")
    logger.info(f"   Filename: {image.filename}")
    logger.info(f"   point_x: {point_x}, point_y: {point_y}")
    logger.info(f"   material: {material}, color_hex: {color_hex}, strength: {strength}")

    if _predictor is None:
        logger.error("❌ SAM-2 not loaded")
        raise HTTPException(503, "SAM-2 not loaded")

    try:
        img_bytes = await image.read()
        image_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h = image_pil.size
        logger.info(f"   Original dimensions: {w}x{h}")
        max_size = 1024
        if w > max_size or h > max_size:
            image_pil.thumbnail((max_size, max_size))
            logger.info(f"   Resized to: {image_pil.size}")
        image_np = np.array(image_pil)

        try:
            if color_hex.startswith("0x"):
                color_hex_int = int(color_hex, 16)
            elif color_hex.startswith("FF"):
                color_hex_int = int("0x" + color_hex, 16)
            else:
                color_hex_int = int(color_hex)
        except ValueError:
            color_hex_int = 0xFF8B4513
        rgb_hex = color_hex_int & 0xFFFFFF

        with torch.no_grad():
            _predictor.set_image(image_np)
            masks, scores, _ = _predictor.predict(
                point_coords=np.array([[int(point_x), int(point_y)]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        mask_area = np.sum(best_mask)
        logger.info(f"   SAM-2: got {len(masks)} masks, best score={scores[best_idx]:.3f}, mask area={mask_area} pixels")

        if mask_area < 10:
            logger.warning("⚠️ Mask area is very small – object might not be detected!")

        mask_binary = (best_mask > 0.5).astype(np.uint8) * 255
        logger.info(f"   Mask white pixels: {np.sum(mask_binary > 0)} of {mask_binary.size}")
        mask_pil = Image.fromarray(mask_binary, mode='L')

        recolor_start = time.time()
        logger.info("   Recoloring with texture preservation...")
        target_r = (rgb_hex >> 16) & 0xFF
        target_g = (rgb_hex >> 8) & 0xFF
        target_b = rgb_hex & 0xFF
        result = recolor_image_preserve_texture(image_pil, mask_pil, target_r, target_g, target_b)
        recolor_time = time.time() - recolor_start
        logger.info(f"   Recoloring took {recolor_time:.2f}s")

        buf = BytesIO()
        result.save(buf, format="PNG")
        total_time = time.time() - start_time
        logger.info(f"✅ Request completed in {total_time:.2f}s total")
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Request failed after {total_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
