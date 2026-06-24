"""
AI-сервер для сегментации + перекраски с SAM-2 и ControlNet Inpaint
Улучшенное логирование для отладки запросов от приложений
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

# ---------- Настройка логирования ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --------------------- Lifespan ---------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _pipe, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {_device}")

    # Загрузка SAM-2
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
        _predictor.model = _predictor.model.to(_device).eval()
        logger.info("✅ SAM-2 Hiera-L loaded")
    except Exception as e:
        logger.error(f"❌ SAM-2 load error: {e}")
        _predictor = None

    # Загрузка ControlNet Inpaint
    try:
        from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint",
            torch_dtype=torch.float32
        )
        _pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None
        ).to(_device)
        _pipe.safety_checker = None
        _pipe.requires_safety_checker = False
        logger.info("✅ ControlNet Inpaint pipeline loaded")
    except Exception as e:
        logger.error(f"❌ ControlNet Inpaint pipeline load error: {e}")
        _pipe = None

    yield

    # Очистка при завершении
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 GPU cache cleared")


# --------------------- FastAPI приложение ---------------------
app = FastAPI(title="AI Colorization API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None
_pipe = None
_device = "cpu"

# --------------------- Промпты ---------------------
# Акцент: ТОЛЬКО изменение цвета — форма, текстура, освещение и перспектива сохраняются

MATERIAL_PROMPTS = {
    "metal": "A bright {color} metal {object}, same shape, same geometry, same metallic reflections, same lighting, same perspective, photorealistic, rich {color} metallic surface, highly detailed",
    "wood": "A rich {color} wooden {object}, same shape, same wood grain texture, same lighting, same perspective, photorealistic, deep {color} wood finish, natural look",
    "plastic": "A vivid {color} plastic {object}, same shape, same smooth glossy surface, same lighting, same perspective, photorealistic, bright {color} color, high quality",
    "fabric": "A vibrant {color} fabric {object}, same shape, same weave texture, same folds, same lighting, same perspective, photorealistic, rich {color} textile, high quality",
    "glass": "A {color} tinted glass {object}, same shape, same transparency, same reflections, same lighting, same perspective, photorealistic, {color} glass, elegant",
    "leather": "A rich {color} leather {object}, same shape, same grain texture, same stitching, same lighting, same perspective, photorealistic, premium {color} leather",
    "ceramic": "A beautiful {color} ceramic {object}, same shape, same glaze finish, same lighting, same perspective, photorealistic, smooth {color} ceramic",
    "concrete": "A {color} concrete {object}, same shape, same rough texture, same lighting, same perspective, photorealistic, {color} concrete surface, industrial look",
}

DEFAULT_PROMPT = "A beautiful {color} {object}, same shape, same texture, same lighting, same perspective, photorealistic, {color} color, highly detailed"

NEGATIVE_PROMPT = (
    "different shape, different object, deformed, distorted, morphed, "
    "changed geometry, new object, replaced object, wrong shape, "
    "wrong color, different color, original color, "
    "blurry, low quality, artifacts, noise, watermark, "
    "extra objects, missing parts, cropped, out of frame"
)


def get_color_hex_name(hex_color: int) -> str:
    """Конвертирует HEX-цвет в читаемое английское название для промпта."""
    from colorsys import rgb_to_hsv
    r = (hex_color >> 16) & 0xFF
    g = (hex_color >> 8) & 0xFF
    b = hex_color & 0xFF
    h, s, v = rgb_to_hsv(r / 255, g / 255, b / 255)

    # Ахроматические цвета (серые тона)
    if s < 0.12:
        if v > 0.85:
            return "white"
        if v < 0.20:
            return "black"
        if v < 0.45:
            return "dark gray"
        return "light gray"

    # Хроматические цвета — по оттенку + яркость/насыщенность
    if h < 0.03 or h > 0.97:
        return "dark red" if v < 0.5 else "red"
    if h < 0.08:
        return "orange" if s > 0.6 else "brown"
    if h < 0.15:
        if v < 0.35:
            return "dark brown"
        if v < 0.60:
            return "brown"
        return "light brown"
    if h < 0.20:
        return "yellow"
    if h < 0.40:
        if v < 0.45:
            return "dark green"
        if s > 0.5:
            return "green"
        return "olive green"
    if h < 0.55:
        return "teal" if s < 0.6 else "green"
    if h < 0.70:
        if v < 0.35:
            return "navy blue"
        if s < 0.45:
            return "light blue"
        return "blue"
    if h < 0.80:
        return "purple"
    if h < 0.92:
        return "pink" if v > 0.70 else "dark red"
    return "red"


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": _device,
        "models_loaded": _predictor is not None and _pipe is not None
    }


@app.post("/ai-recolor")
async def ai_recolor(
    image: UploadFile = File(...),
    point_x: float = Form(...),
    point_y: float = Form(...),
    material: str = Form("wood"),
    color_hex: str = Form("0xFF8B4513"),
    object_name: str = Form("object"),
    strength: float = Form(0.85),
    guidance_scale: float = Form(9.0),
    num_inference_steps: int = Form(20),
):
    start_time = time.time()
    logger.info("📥 ===== NEW REQUEST =====")
    logger.info(f"   Filename: {image.filename}")
    logger.info(f"   point_x: {point_x}, point_y: {point_y}")
    logger.info(f"   object_name: {object_name}, material: {material}, color_hex: {color_hex}, strength: {strength}, guidance_scale: {guidance_scale}, steps: {num_inference_steps}")

    if _predictor is None or _pipe is None:
        logger.error("❌ Models not loaded")
        raise HTTPException(503, "Models not loaded")

    try:
        # 1. Чтение и загрузка изображения
        img_bytes = await image.read()
        logger.info(f"   Image size: {len(img_bytes)} bytes")
        try:
            source_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"❌ PIL decode error: {e}")
            raise HTTPException(400, f"Invalid image: {e}")
        if source_image is None:
            logger.error("❌ Failed to decode image: source_image is None")
            raise HTTPException(400, "Failed to decode image")
        w, h = source_image.size
        logger.info(f"   Original dimensions: {w}x{h}")

        # Ресайз до разумного размера (макс. 1024x1024) для стабильности
        max_size = 1024
        if w > max_size or h > max_size:
            source_image.thumbnail((max_size, max_size))
            if source_image is None:
                logger.error("❌ source_image became None after thumbnail")
                raise HTTPException(500, "Internal error: image resize failed")
            new_w, new_h = source_image.size
            logger.info(f"   Resized to: {new_w}x{new_h}")
        else:
            logger.info("   No resize needed")

        source_image_np = np.array(source_image)
        logger.info(f"   Image array shape: {source_image_np.shape}")

        scale_x = source_image.width / w
        scale_y = source_image.height / h

        logger.info(f"   Resize scale: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

        scaled_point_x = int(point_x * scale_x)
        scaled_point_y = int(point_y * scale_y)
        logger.info(f"   Scaled prompt point: ({point_x}, {point_y}) -> ({scaled_point_x}, {scaled_point_y})")
        point_x = scaled_point_x
        point_y = scaled_point_y

        # 2. Преобразование color_hex
        if color_hex.startswith("0x") or color_hex.startswith("0X"):
            color_hex_int = int(color_hex, 16)
        else:
            color_hex_int = int(color_hex)
        logger.info(f"   Parsed color_hex_int: {color_hex_int}")

        # Проверяем, что source_image всё ещё валидна после всех операций
        logger.info(f"   source_image type before generation: {type(source_image)}, size: {source_image.size if source_image else 'N/A'}")
        if source_image is None:
            logger.error("❌ source_image is None before generation")
            raise HTTPException(500, "Internal error: source_image is None before generation")

        # 3. Сегментация SAM-2
        seg_start = time.time()
        with torch.no_grad():
            _predictor.set_image(source_image_np)
            masks, scores, logits = _predictor.predict(
                point_coords=np.array([[point_x, point_y]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        mask_area = np.sum(best_mask)
        logger.info(
            f"   SAM-2: got {len(masks)} masks, "
            f"best score={scores[best_idx]:.3f}, mask area={mask_area} pixels"
        )

        if mask_area < 10:
            logger.warning("⚠️  Mask area is very small – object might not be detected!")

        seg_time = time.time() - seg_start
        logger.info(f"   Segmentation took {seg_time:.2f}s")

        # 4. Формирование промпта с цветом и названием объекта
        color_name = get_color_hex_name(color_hex_int)
        prompt_template = MATERIAL_PROMPTS.get(material, DEFAULT_PROMPT)
        prompt = prompt_template.format(color=color_name, object=object_name)

        logger.info(f"   object_name: '{object_name}', color_name: '{color_name}'")
        logger.info(f"   Prompt: {prompt}")
        logger.info(f"   Negative prompt: {NEGATIVE_PROMPT}")

        # 5. Создание маски PIL
        mask_pil = Image.fromarray((best_mask * 255).astype(np.uint8), mode='L')
        if mask_pil is None:
            logger.error("❌ mask_pil is None before generation")
            raise HTTPException(500, "Internal error: mask generation failed")

        # Проверяем все переменные перед инференсом
        logger.info(
            f"   Pre-gen check: source_image={type(source_image).__name__}, "
            f"mask_pil={type(mask_pil).__name__}, source_image_np shape={source_image_np.shape}"
        )
        if source_image is None:
            logger.error("❌ source_image is None before _pipe")
            raise HTTPException(500, "Internal error: source_image is None before generation")

        # 6. Инференс
        # Прямое использование strength из запроса с клипом [0.0, 1.0].
        # Без control_image, чтобы промпт имел полный эффект.
        effective_strength = max(0.0, min(1.0, float(strength)))

        gen_start = time.time()
        logger.info(
            f"   Generation params: strength={effective_strength:.2f}, "
            f"guidance_scale={guidance_scale}, steps={num_inference_steps}"
        )
        logger.info("   Generating...")

        result = _pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=source_image_np,
            mask_image=mask_pil,
            control_image=None,
            strength=effective_strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(_device).manual_seed(42),
        ).images[0]

        gen_time = time.time() - gen_start
        logger.info(f"   Generation took {gen_time:.2f}s")

        # 8. Возврат PNG
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