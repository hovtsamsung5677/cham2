"""
AI-сервер для сегментации + перекраски с SAM-2 и FLUX.2 [klein] 4B
Улучшенное логирование для отладки запросов от приложений
"""
import logging
import time
import traceback
import gc
import numpy as np
import torch
from io import BytesIO
from contextlib import asynccontextmanager
from diffusers import Flux2KleinInpaintPipeline
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

    # Загрузка FLUX.2 [klein] 4B (Apache 2.0)
    try:
        _pipe = Flux2KleinInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.bfloat16
        )
        _pipe.to("cuda")
        logger.info("✅ FLUX.2 [klein] 4B loaded")
    except Exception as e:
        logger.error(f"❌ FLUX.2 load error: {e}")
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
# Используем точные имя цветов (CSS4/X11) вместо hex-кодов

MATERIAL_PROMPTS = {
    "metal": "The {object} is recolored to {color} metal, same shape, same geometry, same metallic reflections, same lighting, same perspective, photorealistic, rich {color} metallic surface, highly detailed",
    "wood": "The {object} is recolored to {color} wooden, same shape, same wood grain texture, same lighting, same perspective, photorealistic, deep {color} wood finish, natural look",
    "plastic": "The {object} is recolored to {color} plastic, same shape, same smooth glossy surface, same lighting, same perspective, photorealistic, bright {color} color, high quality",
    "fabric": "The {object} is recolored to {color} fabric, same shape, same weave texture, same folds, same lighting, same perspective, photorealistic, rich {color} textile, high quality",
    "glass": "The {object} is recolored to {color} tinted glass, same shape, same transparency, same reflections, same lighting, same perspective, photorealistic, elegant {color} glass",
    "leather": "The {object} is recolored to {color} leather, same shape, same grain texture, same stitching, same lighting, same perspective, photorealistic, premium {color} leather",
    "ceramic": "The {object} is recolored to {color} ceramic, same shape, same glaze finish, same lighting, same perspective, photorealistic, smooth {color} ceramic",
    "concrete": "The {object} is recolored to {color} concrete, same shape, same rough texture, same lighting, same perspective, photorealistic, industrial {color} concrete surface",
    "bronze": "The {object} is recolored to bright bronze metal, same shape, same geometry, same shiny metallic reflections, same lighting, same perspective, photorealistic, rich bright bronze metallic surface, highly detailed",
}

DEFAULT_PROMPT = "The {object} is recolored to {color}, same shape, same texture, same lighting, same perspective, photorealistic, beautiful {color} color, highly detailed"

BRIGHTNESS_MODIFIERS = {
    "very dark": (0.0, 0.25),
    "dark": (0.25, 0.40),
    "medium": (0.40, 0.60),
    "bright": (0.60, 0.80),
    "very bright": (0.80, 1.0),
}

NEGATIVE_PROMPT = (
    "different shape, different object, deformed, distorted, morphed, "
    "changed geometry, new object, replaced object, wrong shape, "
    "wrong color, different color, original color, old color, previous color, "
    "blurry, low quality, artifacts, noise, watermark, "
    "extra objects, missing parts, cropped, out of frame"
)


# CSS4/X11 именованные цвета с точными диапазонами для максимальной точности
# Формат: (h_min, h_max, s_min, s_max, v_min, v_max, name)
_NAMED_COLORS = [
    # Красный семейство
    (0.97, 1.0, 0.0, 1.0, 0.0, 0.15, "dark red"),
    (0.97, 1.0, 0.0, 1.0, 0.15, 0.25, "maroon"),
    (0.97, 1.0, 0.0, 1.0, 0.25, 0.35, "brown red"),
    (0.97, 1.0, 0.0, 1.0, 0.35, 0.45, "firebrick"),
    (0.97, 1.0, 0.0, 1.0, 0.45, 0.55, "crimson"),
    (0.97, 1.0, 0.0, 1.0, 0.55, 0.65, "indian red"),
    (0.97, 1.0, 0.0, 1.0, 0.65, 0.75, "salmon"),
    (0.97, 1.0, 0.0, 1.0, 0.75, 0.85, "light coral"),
    (0.97, 1.0, 0.0, 1.0, 0.85, 1.0, "red"),
    # Красный (нижняя часть круга оттенков)
    (0.0, 0.03, 0.0, 1.0, 0.0, 0.15, "dark red"),
    (0.0, 0.03, 0.0, 1.0, 0.15, 0.25, "maroon"),
    (0.0, 0.03, 0.0, 1.0, 0.25, 0.35, "brown red"),
    (0.0, 0.03, 0.0, 1.0, 0.35, 0.45, "firebrick"),
    (0.0, 0.03, 0.0, 1.0, 0.45, 0.55, "crimson"),
    (0.0, 0.03, 0.0, 1.0, 0.55, 0.65, "indian red"),
    (0.0, 0.03, 0.0, 1.0, 0.65, 0.75, "salmon"),
    (0.0, 0.03, 0.0, 1.0, 0.75, 0.85, "light coral"),
    (0.0, 0.03, 0.0, 1.0, 0.85, 1.0, "red"),
    
    # Оранжевый/коричневый
    (0.08, 0.12, 0.4, 1.0, 0.5, 0.75, "bronze"),
    (0.08, 0.12, 0.6, 1.0, 0.6, 1.0, "orange"),
    (0.08, 0.12, 0.5, 1.0, 0.4, 0.5, "brown orange"),
    (0.08, 0.12, 0.0, 1.0, 0.35, 0.55, "brown"),
    
    # Жёлтый/золотой
    (0.12, 0.22, 0.7, 1.0, 0.75, 1.0, "gold"),
    (0.12, 0.22, 0.6, 1.0, 0.65, 0.75, "goldenrod"),
    (0.12, 0.22, 0.5, 1.0, 0.55, 0.65, "amber"),
    (0.12, 0.22, 0.4, 1.0, 0.45, 0.55, "yellow ochre"),
    (0.12, 0.22, 0.0, 1.0, 0.5, 0.6, "olive"),
    (0.12, 0.22, 0.0, 1.0, 0.6, 1.0, "yellow"),
    
    # Зелёный
    (0.22, 0.35, 0.7, 1.0, 0.45, 1.0, "forest green"),
    (0.22, 0.35, 0.6, 1.0, 0.35, 0.45, "dark green"),
    (0.22, 0.35, 0.5, 1.0, 0.45, 0.55, "seagreen"),
    (0.22, 0.35, 0.5, 1.0, 0.55, 0.65, "green"),
    (0.22, 0.35, 0.0, 1.0, 0.65, 0.75, "olive green"),
    
    # Бирюзовый/циан
    (0.35, 0.40, 0.7, 1.0, 0.55, 1.0, "teal"),
    (0.40, 0.50, 0.4, 1.0, 0.65, 1.0, "cyan"),
    (0.35, 0.50, 0.0, 1.0, 0.75, 1.0, "aquamarine"),
    
    # Синий
    (0.50, 0.55, 0.7, 1.0, 0.45, 1.0, "dark cyan"),
    (0.50, 0.60, 0.0, 1.0, 0.75, 1.0, "aqua"),
    (0.60, 0.70, 0.0, 1.0, 0.35, 0.45, "navy blue"),
    (0.60, 0.70, 0.5, 1.0, 0.45, 0.55, "steel blue"),
    (0.60, 0.70, 0.0, 1.0, 0.60, 0.70, "light blue"),
    (0.60, 0.70, 0.6, 1.0, 0.55, 0.65, "royal blue"),
    (0.60, 0.70, 0.7, 1.0, 0.65, 1.0, "blue"),
    
    # Фиолетовый
    (0.70, 0.75, 0.6, 1.0, 0.55, 1.0, "violet"),
    (0.70, 0.80, 0.5, 1.0, 0.45, 0.55, "slate blue"),
    (0.70, 0.80, 0.0, 1.0, 0.55, 0.75, "purple"),
    
    # Розовый
    (0.80, 0.85, 0.7, 1.0, 0.70, 1.0, "hot pink"),
    (0.80, 0.92, 0.6, 1.0, 0.60, 0.70, "deep pink"),
    (0.80, 0.92, 0.5, 1.0, 0.50, 0.60, "pink"),
    (0.80, 0.92, 0.0, 1.0, 0.60, 0.85, "light pink"),
    (0.80, 0.92, 0.0, 1.0, 0.35, 0.40, "dark red"),
]

# Серые оттенки (по значению value)
_GRAY_COLORS = [
    (0.90, "white"),
    (0.75, "off white"),
    (0.65, "light gray"),
    (0.55, "silver"),
    (0.45, "dark gray"),
    (0.35, "gray"),
    (0.25, "dim gray"),
    (0.10, "black"),
]


def get_color_hex_name(hex_color: int) -> str:
    """Конвертирует HEX-цвет в точное читаемое английское название для промпта.
    Использует 50+ именованных цветов CSS4/X11 с учётом hue, saturation и value."""
    from colorsys import rgb_to_hsv
    r = (hex_color >> 16) & 0xFF
    g = (hex_color >> 8) & 0xFF
    b = hex_color & 0xFF
    h, s, v = rgb_to_hsv(r / 255, g / 255, b / 255)
    
    # Ахроматические цвета (низкая насыщенность)
    if s < 0.12:
        for threshold, name in _GRAY_COLORS:
            if v >= threshold:
                return name
        return "black"
    
    # Хроматические цвета — проверяем диапазоны по hue/sat/value
    for h_min, h_max, s_min, s_max, v_min, v_max, name in _NAMED_COLORS:
        # Для красного: проверяем wrap-around (h > 0.97 или h < 0.03)
        hue_match = (h_min <= h <= h_max) or (h_min > h_max and (h >= h_min or h <= h_max))
        if hue_match and s_min <= s <= s_max and v_min <= v <= v_max:
            return name
    
    # Fallback — возвращаем базовое название по hue с модификаторами яркости
    hue_positions = [
        (0.0, "red"), (0.08, "orange"), (0.16, "yellow"), 
        (0.25, "chartreuse"), (0.35, "green"), (0.50, "spring green"),
        (0.60, "cyan"), (0.70, "blue"), (0.80, "violet"), (0.90, "magenta")
    ]
    for base_hue, base_name in hue_positions:
        # Обрабатываем wrap-around для красного
        if (base_name == "red" and h > 0.95):
            base_hue_adj = 1.0
        else:
            base_hue_adj = h
        if abs(base_hue_adj - base_hue) < 0.05:
            # Добавляем точный модификатор яркости
            if v < 0.3:
                return f"very dark {base_name}"
            elif v < 0.45:
                return f"dark {base_name}"
            elif v < 0.60:
                return f"medium {base_name}"
            elif v < 0.75:
                return f"bright {base_name}"
            else:
                return f"vivid {base_name}"
    
    return "unknown color"


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
    strength: float = Form(1.0),
    guidance_scale: float = Form(1.0),
    num_inference_steps: int = Form(4),
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
            if hasattr(_predictor, 'reset_state'):
                _predictor.reset_state()
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

        # 4. Формирование промпта с цветом (именованное название) и названием объекта
        color_name = get_color_hex_name(color_hex_int)
        hex_color_str = f"#{color_hex_int:06x}"
        
        # Яркие цвета не нужно усиливать словом "bright"
        bright_colors = {"light blue", "light coral", "light pink", "white", "off white", "silver", "yellow", "aqua", "cyan", "light gray"}
        if color_name in bright_colors:
            prompt_template = MATERIAL_PROMPTS.get(material, DEFAULT_PROMPT).replace("bright ", "").replace("vivid ", "")
        else:
            prompt_template = MATERIAL_PROMPTS["bronze"] if (material == "metal" and color_name == "bronze") else MATERIAL_PROMPTS.get(material, DEFAULT_PROMPT)
        prompt = prompt_template.format(color=color_name, object=object_name)

        logger.info(f"   object_name: '{object_name}', color_name: '{color_name}', color_hex: '{hex_color_str}'")
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

        # 6. Инференс с FLUX.2 [klein] 4B
        logger.info(f"    source_image type: {type(source_image)}, size: {source_image.size}")
        if source_image is None:
            logger.error("❌ source_image is None before generation")
            raise HTTPException(500, "source_image is None before generation")
        if mask_pil is None:
            logger.error("❌ mask_pil is None before generation")
            raise HTTPException(500, "mask_pil is None before generation")

        # Для FLUX inpainting фиксируем 8 шагов
        effective_steps = 8
        effective_guidance = guidance_scale if guidance_scale > 0 else 5.0
        effective_strength = strength if strength is not None else 0.85

        gen_start = time.time()
        logger.info(
            f"   Generation params: guidance_scale={effective_guidance}, steps={effective_steps}, strength={effective_strength}"
        )
        logger.info("   Generating...")

        try:
            result = _pipe(
                image=source_image,
                mask_image=mask_pil,
                prompt=prompt,
                guidance_scale=effective_guidance,
                num_inference_steps=effective_steps,
                strength=effective_strength,
                generator=torch.Generator(_device).manual_seed(int(time.time() * 1000) % (2**32)),
            ).images[0]
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"❌ CUDA OOM: {e}. Try lower resolution or enable_sequential_cpu_offload()")
            raise HTTPException(500, "GPU out of memory. Try lowering the image resolution.")
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            raise HTTPException(500, f"Generation failed: {e}")

        gen_time = time.time() - gen_start
        logger.info(f"   Generation took {gen_time:.2f}s")

        # 8. Возврат PNG
        buf = BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        total_time = time.time() - start_time
        logger.info(f"✅ Request completed in {total_time:.2f}s total")

        # Очистка памяти после обработки
        del source_image_np
        del masks, scores, logits, best_mask
        del source_image, mask_pil, result

        response_content = buf.getvalue()
        buf.close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        return Response(content=response_content, media_type="image/png")

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Request failed after {total_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)