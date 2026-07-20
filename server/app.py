"""
AI-сервер для сегментации + перекраски с SAM-2 и FLUX.2 [klein] 4B
Улучшенное логирование для отладки запросов от приложений
"""
import asyncio
import logging
import time
import traceback
import gc
import math
import os
import re
import secrets
import numpy as np
import torch
from io import BytesIO
from contextlib import asynccontextmanager
from diffusers import Flux2KleinInpaintPipeline
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Загрузка переменных окружения из .env (если файл есть и пакет установлен)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- Настройка логирования ----------
# Уровень логирования управляется переменной окружения LOG_LEVEL.
# В продакшене рекомендуется WARNING, чтобы не писать в логи имена файлов,
# координаты и прочие потенциально чувствительные данные.
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ---------- Ограничения безопасности ----------
# Максимальный размер загружаемого изображения (20 MB)
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024

# Максимальные размеры изображения в пикселях (для валидации координат клика)
MAX_IMAGE_DIMENSION = 8192

# Допустимые материалы (значения, приходящие с клиента)
ALLOWED_MATERIALS = {
    "metal", "wood", "plastic", "fabric", "glass",
    "leather", "ceramic", "concrete", "bronze",
    "no_texture",
}

# Разрешённые источники (CORS). Задаются через переменную окружения
# ALLOWED_ORIGINS в виде списка через запятую. Пустой список = запросы
# из браузера с других доменов запрещены (нативному приложению CORS не нужен).
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]

# API-ключи для доступа к серверу. Задаются через переменную окружения
# API_KEYS в виде списка через запятую. Если список пуст — проверка
# отключена (например, на локальной машине для отладки).
_api_keys_env = os.getenv("API_KEYS", "").strip()
VALID_API_KEYS = {k.strip() for k in _api_keys_env.split(",") if k.strip()}

# Rate limiter: ограничение количества запросов с одного IP
limiter = Limiter(key_func=get_remote_address)

# Семафор ограничивает количество одновременных GPU-задач (инференс SAM-2 + FLUX).
# На одной видеокарте тяжёлый инференс всё равно сериализуется, а семафор защищает
# модели от конкурентного доступа и предотвращает CUDA OOM при одновременном
# выделении памяти под несколько больших батчей. Сколько запросов могут
# одновременно "заходить" в инференс — задаётся переменной окружения MAX_CONCURRENT_JOBS
# (по умолчанию 2). Остальные запросы параллельно ждут в очереди, не блокируя event loop.
_MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
_inference_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JOBS)


def sanitize_prompt_text(text: str, max_length: int = 50) -> str:
    """Очищает пользовательскую строку для безопасной вставки в промпт.
    Оставляет только буквы, цифры, пробелы и дефис; обрезает по длине."""
    if not text:
        return "object"
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    cleaned = cleaned[:max_length]
    return cleaned or "object"


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)

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

# Подключение rate limiter к приложению
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=bool(ALLOWED_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

_predictor = None
_pipe = None
_device = "cpu"


# --------------------- Проверка API-ключа ---------------------
def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Проверяет API-ключ из заголовка X-API-Key.
    Если серверные ключи не заданы (VALID_API_KEYS пуст), проверка пропускается."""
    if not VALID_API_KEYS:
        return None
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        logger.warning("⛔ Rejected request with invalid/missing API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key

# --------------------- Промпты ---------------------
# Акцент: ТОЛЬКО изменение цвета — форма, текстура, освещение и перспектива сохраняются
# Используем точные имя цветов (CSS4/X11) вместо hex-кодов

MATERIAL_PROMPTS = {
    "metal": "The {object} is recolored to {color} metal, same shape, same geometry, same metallic reflections, same lighting, same perspective, photorealistic, rich {color} metallic surface, highly detailed",
    "silver": "The {object} is recolored to polished silver metal, same shape, same geometry, same bright metallic reflections, same lighting, same perspective, photorealistic, mirror-like {color} silver metallic surface, highly reflective",
    "stainless_steel": "The {object} is recolored to brushed stainless steel, same shape, same geometry, same metallic reflections, same lighting, same perspective, photorealistic, clean {color} stainless steel with subtle brushed texture, reflective surface",
    "gold": "The {object} is recolored to shiny gold metal, same shape, same geometry, same bright metallic reflections, same lighting, same perspective, photorealistic, lustrous {color} golden metallic surface, highly reflective",
    "bronze": "The {object} is recolored to bright bronze metal, same shape, same geometry, same shiny metallic reflections, same lighting, same perspective, photorealistic, rich bright bronze metallic surface, highly detailed",
    "brass": "The {object} is recolored to brass metal, same shape, same geometry, same yellow metallic reflections, same lighting, same perspective, photorealistic, warm {color} brass metallic surface, highly reflective",
    "copper": "The {object} is recolored to copper metal, same shape, same geometry, same reddish metallic reflections, same lighting, same perspective, photorealistic, rich {color} copper metallic surface with warm tone, highly detailed",
    "titanium": "The {object} is recolored to deep gray-steel titanium metal, same shape, same geometry, same deep gray-steel metallic reflections, same lighting, same perspective, photorealistic, smooth deep gray-steel metallic surface, highly reflective",
    "wood": "The {object} is recolored to {color} wooden, same shape, same wood grain texture, same lighting, same perspective, photorealistic, deep {color} wood finish, natural look",
    "plastic": "The {object} is recolored to {color} plastic, same shape, same smooth glossy surface, same lighting, same perspective, photorealistic, bright {color} color, high quality",
    "fabric": "The {object} is recolored to {color} fabric, same shape, same weave texture, same folds, same lighting, same perspective, photorealistic, rich {color} textile, high quality",
    "glass": "The {object} is recolored to {color} tinted glass, same shape, same transparency, same reflections, same lighting, same perspective, photorealistic, elegant {color} glass",
    "leather": "The {object} is recolored to {color} leather, same shape, same grain texture, same stitching, same lighting, same perspective, photorealistic, premium {color} leather",
    "ceramic": "The {object} is recolored to {color} ceramic, same shape, same glaze finish, same lighting, same perspective, photorealistic, smooth {color} ceramic",
    "concrete": "The {object} is recolored to {color} concrete, same shape, same rough texture, same lighting, same perspective, photorealistic, industrial {color} concrete surface",
    "no_texture": "The {object} is recolored to {color}, same shape, flat {color} color, no texture, smooth matte surface, photorealistic, solid {color} color, clean finish",
  }

DEFAULT_PROMPT = "The {object} is recolored to {color}, same shape, matching the requested material, same lighting, same perspective, photorealistic, beautiful {color} color, highly detailed"

BRIGHTNESS_MODIFIERS = {
    "very dark": (0.0, 0.25),
    "dark": (0.25, 0.40),
    "medium": (0.40, 0.60),
    "bright": (0.60, 0.80),
    "very bright": (0.80, 1.0),
}

# CSS4/X11 таблица цветов для поиска ближайшего имени
# Формат: ((R, G, B), name)
_CSS_NAMED_COLORS = [
    # Красный
    ((255, 0, 0), "red"),
    ((220, 20, 60), "crimson"),
    ((255, 0, 0), "red"),
    ((128, 0, 0), "maroon"),
    ((178, 34, 34), "firebrick"),
    ((139, 0, 0), "dark red"),
    ((165, 42, 42), "brown"),
    ((178, 34, 34), "firebrick"),
    ((205, 92, 92), "indian red"),
    ((240, 128, 128), "light coral"),
    ((250, 128, 114), "salmon"),
    ((255, 99, 71), "tomato"),
    ((255, 69, 0), "orange red"),
    
    # Оранжевый/коричневый
    ((255, 140, 0), "dark orange"),
    ((255, 165, 0), "orange"),
    ((210, 105, 30), "chocolate"),
    ((139, 69, 19), "saddle brown"),
    ((160, 82, 45), "sienna"),
    ((205, 133, 63), "peru"),
    ((222, 184, 135), "burlywood"),
    ((244, 164, 96), "sandy brown"),
    ((184, 134, 11), "dark goldenrod"),
    
    # Жёлтый/золотой
    ((255, 215, 0), "gold"),
    ((218, 165, 32), "goldenrod"),
    ((255, 223, 0), "gold"),
    ((189, 183, 107), "dark khaki"),
    ((240, 230, 140), "khaki"),
    ((255, 250, 205), "lemon chiffon"),
    ((255, 255, 0), "yellow"),
    ((154, 205, 50), "yellow green"),
    ((128, 128, 0), "olive"),
    
    # Зелёный
    ((0, 128, 0), "green"),
    ((0, 100, 0), "dark green"),
    ((34, 139, 34), "forest green"),
    ((107, 142, 35), "olive green"),
    ((50, 205, 50), "lime green"),
    ((144, 238, 144), "light green"),
    ((0, 255, 0), "lime"),
    ((60, 179, 113), "medium sea green"),
    ((46, 139, 87), "sea green"),
    
    # Бирюзовый/циан
    ((32, 178, 170), "light sea green"),
    ((0, 206, 209), "dark turquoise"),
    ((64, 224, 208), "turquoise"),
    ((0, 255, 255), "cyan"),
    ((175, 238, 238), "pale turquoise"),
    ((127, 255, 212), "aquamarine"),
    ((0, 128, 128), "teal"),
    
    # Синий
    ((0, 191, 255), "deep sky blue"),
    ((135, 206, 235), "sky blue"),
    ((70, 130, 180), "steel blue"),
    ((95, 158, 160), "cadet blue"),
    ((100, 149, 237), "cornflower blue"),
    ((30, 144, 255), "dodger blue"),
    ((65, 105, 225), "royal blue"),
    ((0, 0, 255), "blue"),
    ((0, 0, 205), "medium blue"),
    ((0, 0, 139), "navy blue"),
    ((25, 25, 112), "midnight blue"),
    
    # Фиолетовый
    ((72, 61, 139), "dark slate blue"),
    ((106, 90, 205), "slate blue"),
    ((123, 104, 238), "medium slate blue"),
    ((138, 43, 226), "blue violet"),
    ((148, 0, 211), "dark violet"),
    ((75, 0, 130), "indigo"),
    ((153, 50, 204), "dark orchid"),
    ((186, 85, 211), "medium orchid"),
    ((0, 0, 128), "navy"),
    ((238, 130, 238), "violet"),
    
    # Розовый/пурпурный
    ((255, 0, 255), "magenta"),
    ((199, 21, 133), "medium violet red"),
    ((219, 112, 147), "pale violet red"),
    ((255, 20, 147), "deep pink"),
    ((255, 105, 180), "hot pink"),
    ((255, 192, 203), "pink"),
    ((255, 182, 193), "light pink"),
    ((255, 0, 255), "fuchsia"),
    ((221, 160, 221), "plum"),
    ((238, 130, 238), "violet"),
    
    # Коричневый
    ((165, 42, 42), "brown"),
    ((139, 69, 19), "saddle brown"),
    ((160, 82, 45), "sienna"),
    ((210, 105, 30), "chocolate"),
    ((205, 133, 63), "peru"),
    ((205, 127, 50), "bronze"),  # Цвет бронзы
    ((222, 184, 135), "burlywood"),
    ((244, 164, 96), "sandy brown"),
    
    # Специальные металлы
    ((201, 166, 107), "brass"),  # Латунь (0xFFC9A66B)
    ((205, 127, 50), "copper"),  # Медь (0xFFCD7F32)
    
    # Серые
    ((192, 192, 192), "silver"),
    ((211, 211, 211), "light gray"),
    ((119, 136, 153), "light slate gray"),
    ((105, 105, 105), "dim gray"),
    ((250, 250, 250), "snow"),
    ((28, 28, 28), "dim gray"),
    ((0, 77, 64), "dark green"),
    ((93, 64, 55), "dark brown"),
    ((62, 39, 35), "espresso"),
    ((44, 62, 80), "charcoal blue"),
]

# Серые оттенки (по значению value)
_GRAY_COLORS = [
    (0.92, "white"),
    (0.80, "off white"),
    (0.72, "light gray"),
    (0.58, "silver"),
    (0.42, "dark gray"),
    (0.28, "gray"),
    (0.12, "dim gray"),
    (0.04, "black"),
  ]


def get_color_hex_name(hex_color: int) -> str:
    """Конвертирует HEX-цвет в точное читаемое английское название для промпта.
    Использует lookup 50+ цветов CSS4/X11 с поиском ближайшего в RGB-пространстве."""
    r = (hex_color >> 16) & 0xFF
    g = (hex_color >> 8) & 0xFF
    b = hex_color & 0xFF
    mx = max(r, g, b)
    mn = min(r, g, b)
    sat = 0.0 if mx == 0 else (mx - mn) / mx

    # Серые оттенки
    if sat < 0.12:
        val = mx / 255.0
        # Специальные цвета металлов (светло-серые)
        exact_metal_grays = {
            (232, 236, 239): "stainless_steel",  # Нержавейка (0xFFE8ECEF)
            (224, 224, 224): "silver",  # Серебро (0xFFE0E0E0)
            (110, 116, 120): "titanium",  # Титан (0xFF6E7478) - глубокий серо-стальной
        }
        if (r, g, b) in exact_metal_grays:
            return exact_metal_grays[(r, g, b)]
        exact_grays = {
            (255, 255, 255): "white",
            (128, 128, 128): "gray",
            (211, 211, 211): "light gray",
            (169, 169, 169): "dark gray",
            (105, 105, 105): "dim gray",
        }
        if (r, g, b) in exact_grays:
            return exact_grays[(r, g, b)]
        for threshold, name in _GRAY_COLORS:
            if val >= threshold:
                return name
        return "black"

    # Находим ближайший цвет из таблицы по евклидову расстоянию в RGB
    best_name = _CSS_NAMED_COLORS[0][1]
    best_dist = float('inf')
    for (cr, cg, cb), name in _CSS_NAMED_COLORS:
        dr = r - cr
        dg = g - cg
        db = b - cb
        dist = dr*dr + dg*dg + db*db
        if dist < best_dist:
            best_dist = dist
            best_name = name

    exact_names = {
        "black", "white", "red", "green", "blue", "yellow", "cyan", "magenta",
        "orange", "purple", "pink", "brown", "gray", "maroon", "olive", "teal",
        "navy blue", "midnight blue", "dark red", "dark green", "dark blue",
        "light blue", "light green", "light pink", "light coral", "dim gray",
        "dark gray", "light gray", "off white", "silver", "snow",
        "lime", "aqua", "crimson", "firebrick", "indian red", "salmon", "tomato",
        "gold", "goldenrod", "khaki", "lemon chiffon", "yellow green",
        "dark olive green", "forest green", "olive green", "lime green",
        "light sea green", "dark turquoise", "turquoise", "cyan",
        "pale turquoise", "aquamarine", "teal",
        "deep sky blue", "sky blue", "steel blue", "cornflower blue", "dodger blue",
        "royal blue", "medium blue", "navy",
        "dark slate blue", "slate blue", "medium slate blue", "blue violet",
        "dark violet", "indigo", "dark orchid", "medium orchid", "violet",
        "magenta", "medium violet red", "pale violet red", "deep pink",
        "hot pink", "pink", "light pink", "fuchsia", "plum",
        "saddle brown", "sienna", "chocolate", "peru", "burlywood", "sandy brown",
        "dark goldenrod", "dark khaki", "dark green", "dark brown", "espresso",
        "charcoal blue", "light slate gray",
        "stainless_steel", "bronze",
    }
    if best_name in exact_names or best_dist < 2500:
        return best_name

    val = mx / 255.0
    if val < 0.30:
        return "very dark " + best_name
    elif val < 0.45:
        return "dark " + best_name
    elif val > 0.80:
        return "bright " + best_name
    return best_name


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": _device,
        "models_loaded": _predictor is not None and _pipe is not None
    }


def run_recolor_job(
    img_bytes: bytes,
    point_x: float,
    point_y: float,
    material: str,
    color_hex: str,
    color_name: str,
    object_name: str,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    patina: bool,
    color_r: int | None = None,
    color_g: int | None = None,
    color_b: int | None = None,
) -> bytes:
    """Синхронная тяжёлая обработка одного запроса (декод, SAM-2, FLUX, кодирование PNG).

    Вызывается из эндпоинта через ``run_in_threadpool``, чтобы не блокировать
    event loop и позволить FastAPI обрабатывать несколько запросов одновременно.
    Защищена семафором ``_inference_semaphore`` на стороне вызова, поэтому доступ
    к общим моделям ``_predictor``/``_pipe`` сериализован.
    """
    start_time = time.time()

    # 1. Декодирование изображения
    try:
        source_image = Image.open(BytesIO(img_bytes))
        source_image = ImageOps.exif_transpose(source_image)
        source_image = source_image.convert("RGB")
    except Exception as e:
        logger.error(f"❌ PIL decode error: {e}")
        raise HTTPException(400, f"Invalid image: {e}")
    if source_image is None:
        logger.error("❌ Failed to decode image: source_image is None")
        raise HTTPException(400, "Failed to decode image")
    w, h = source_image.size
    logger.info(f"   Image dimensions: {w}x{h} (EXIF orientation applied server-side)")

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
    image_height, image_width = source_image_np.shape[:2]
    image_area = image_width * image_height

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

    # 3. Сегментация SAM-2 с consistency-check (5 прогонов с джиттер-точками)
    seg_start = time.time()

    # Генерируем 4 джиттер-точки вокруг исходной точки клика
    jitter_offsets = [(8, 0), (-8, 0), (0, 8), (0, -8)]
    jitter_points = []
    for dx, dy in jitter_offsets:
        jx = max(0, min(image_width - 1, point_x + dx))
        jy = max(0, min(image_height - 1, point_y + dy))
        jitter_points.append((jx, jy))

    # Выполняем сегментацию для каждой точки
    all_mask_candidates = []
    all_scores = []
    all_coords = [(point_x, point_y)] + jitter_points

    with torch.no_grad():
        if hasattr(_predictor, 'reset_state'):
            _predictor.reset_state()
        _predictor.set_image(source_image_np)

        for cx, cy in all_coords:
            masks, scores, logits = _predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            best_idx_local = np.argmax(scores)
            all_mask_candidates.append(masks[best_idx_local])
            all_scores.append(scores[best_idx_local])
            logger.info(f"   SAM-2: point=({cx}, {cy}), best mask score={scores[best_idx_local]:.3f}")

    # Находим финальную маску по максимальному среднему IoU
    best_mask_idx = 0
    best_mean_iou = 0.0
    for i, mask_i in enumerate(all_mask_candidates):
        ious = [mask_iou(mask_i, all_mask_candidates[j]) for j in range(len(all_mask_candidates)) if j != i]
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_mask_idx = i

    best_mask = all_mask_candidates[best_mask_idx]
    mask_area = np.sum(best_mask)
    mask_area_percent = mask_area / (image_width * image_height) * 100

    logger.info(f"   SAM-2: final mask from point {all_coords[best_mask_idx]}, score={all_scores[best_mask_idx]:.3f}")
    logger.info(f"   SAM-2: mask area={mask_area} pixels ({mask_area_percent:.2f}% of image), mean IoU={best_mean_iou:.3f}")

    # Проверка стабильности сегментации
    if best_mean_iou < 0.5:
        logger.warning(f"⚠️  Low mask consistency (mean IoU={best_mean_iou:.3f} < 0.5) — point may be near object boundary")

    if mask_area < 10:
        logger.warning("⚠️  Mask area is very small – object might not be detected!")

    seg_time = time.time() - seg_start
    logger.info(f"   Segmentation took {seg_time:.2f}s (5 runs with consistency-check)")

    # 4. Формирование промпта с цветом (именованное название) и названием объекта
    # Используем переданное имя цвета если оно есть
    if color_name and color_name != "":
        color_name = color_name
    else:
        color_name = get_color_hex_name(color_hex_int)
    hex_color_str = f"#{color_hex_int:06x}"

    # Точное описание цвета (hex + RGB), чтобы модель получала больше данных
    # о реальном оттенке, особенно когда цвет пришёл из пипетки.
    if None not in (color_r, color_g, color_b):
        exact_color_desc = f"{hex_color_str} (RGB {color_r}, {color_g}, {color_b})"
    else:
        exact_color_desc = hex_color_str

    # Яркие цвета не нужно усиливать словом "bright"
    bright_colors = {"light blue", "light coral", "light pink", "white", "off white", "yellow", "aqua", "cyan", "light gray"}

    # Специальные имена металлов
    exact_metal_names = {"gold", "silver", "bronze", "stainless_steel", "brass", "copper", "titanium"}

    # Flat-matte только когда материал реально «без текстуры».
    # Для остальных материалов всегда используем шаблон материала
    # (с блеском у металлов и текстурой у дерева/кожи/ткани и т.п.),
    # независимо от того, выбран ли вариант текстуры.
    if material == "no_texture":
        # Без текстуры - только цвет (для всех материалов)
        prompt = f"The {object_name} is recolored to {exact_color_desc}, same shape, flat {color_name} color, hex {hex_color_str}, no texture, smooth matte surface, photorealistic"
    elif material == "metal":
        # Металл: блеск и отражения. Конкретный металл берётся по имени цвета,
        # иначе — универсальный металл. Материал «металл» здесь главный.
        if color_name in exact_metal_names:
            prompt_template = MATERIAL_PROMPTS.get(color_name, MATERIAL_PROMPTS["metal"])
        elif color_name in bright_colors:
            prompt_template = MATERIAL_PROMPTS["metal"].replace("bright ", "").replace("vivid ", "")
        else:
            prompt_template = MATERIAL_PROMPTS["bronze"] if color_name == "bronze" else MATERIAL_PROMPTS["metal"]
        prompt = prompt_template.format(color=exact_color_desc, object=object_name)
    else:
        # Любой другой материал (дерево, пластик, ткань, кожа, стекло, керамика, бетон):
        # используем шаблон выбранного материала, цвет задаётся именем color_name.
        # Материал имеет приоритет над тем, как назван цвет, чтобы, например,
        # коричневый или серебристый цвет не превращал дерево/пластик в металл.
        prompt_template = MATERIAL_PROMPTS.get(material, DEFAULT_PROMPT)
        if color_name in bright_colors:
            prompt_template = prompt_template.replace("bright ", "").replace("vivid ", "")
        prompt = prompt_template.format(color=exact_color_desc, object=object_name)

    # Эффект старения (патина) для металла: добавляем признаки износа/окисления
    if material == "metal" and patina:
        prompt += ", with aged patina finish, weathered oxidation, antique worn metal, subtle verdigris and brown patina, realistic aging, uneven discolored surface"

    logger.info(f"   object_name: '{object_name}', color_name: '{color_name}', color_hex: '{hex_color_str}'")
    logger.info(f"   Prompt: {prompt}")

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

    # Используем параметры из запроса с разумными ограничениями.
    # Примечание: Flux2KleinInpaintPipeline — guidance-distilled модель; для неё
    # guidance_scale > 1.0 игнорируется (см. do_classifier_free_guidance в исходниках).
    # Поэтому здесь значение передаётся «как есть», но реального эффекта при distilled=True не даёт.
    effective_steps = min(50, int(num_inference_steps))
    effective_guidance = guidance_scale if guidance_scale > 0 else 1.0
    effective_strength = strength if strength is not None else 1.0

    gen_start = time.time()
    logger.info(
        f"   Generation params: guidance_scale={effective_guidance}, steps={effective_steps}, strength={effective_strength}, prompt='{prompt}'"
    )
    logger.info(f"🎨 Running FLUX.2 inference: steps={effective_steps}, guidance={effective_guidance}, strength={effective_strength}, image_size={source_image.size}")

    try:
        result = _pipe(
            image=source_image,
            mask_image=mask_pil,
            prompt=prompt,
            guidance_scale=effective_guidance,
            num_inference_steps=effective_steps,
            strength=effective_strength,
            generator=torch.Generator(_device).manual_seed(secrets.randbelow(2**32)),
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

    return response_content


@app.post("/ai-recolor")
@limiter.limit("10/minute")
async def ai_recolor(
    request: Request,
    image: UploadFile = File(...),
    point_x: float = Form(...),
    point_y: float = Form(...),
    material: str = Form("wood"),
    color_hex: str = Form("0xFF8B4513"),
    color_name: str = Form(""),
    color_r: int | None = Form(None),
    color_g: int | None = Form(None),
    color_b: int | None = Form(None),
    object_name: str = Form("object"),
    strength: float = Form(1.0),
    guidance_scale: float = Form(5.0),
    num_inference_steps: int = Form(30),
    patina: bool = Form(False),
    api_key: str | None = Depends(verify_api_key),
):
    start_time = time.time()
    logger.info("📥 ===== NEW REQUEST =====")
    logger.debug(f"   Filename: {image.filename}")
    logger.debug(f"   point_x: {point_x}, point_y: {point_y}")
    logger.debug(f"   object_name: {object_name}, material: {material}, color_hex: {color_hex}, color_name: {color_name}, color_rgb: ({color_r}, {color_g}, {color_b}), strength: {strength}, guidance_scale: {guidance_scale}, steps: {num_inference_steps}, patina: {patina}")

    # --- Валидация входящих параметров (защита от аномальных значений/DoS) ---
    # Отсекаем NaN/Infinity
    for _name, _val in (
        ("point_x", point_x), ("point_y", point_y),
        ("strength", strength), ("guidance_scale", guidance_scale),
    ):
        if not math.isfinite(_val):
            raise HTTPException(400, f"Parameter '{_name}' must be a finite number")

    # Координаты клика: неотрицательные и в разумных пределах
    if not (0 <= point_x <= MAX_IMAGE_DIMENSION) or not (0 <= point_y <= MAX_IMAGE_DIMENSION):
        raise HTTPException(400, "point_x/point_y out of allowed range")

    # strength ограничиваем диапазоном [0.1, 1.0]
    strength = float(min(1.0, max(0.1, strength)))

    # guidance_scale ограничиваем диапазоном [1.0, 20.0]
    guidance_scale = float(min(20.0, max(1.0, guidance_scale)))

    # num_inference_steps ограничиваем диапазоном [6, 50]
    num_inference_steps = int(min(50, max(6, num_inference_steps)))

    # Материал должен быть из разрешённого набора
    if material not in ALLOWED_MATERIALS:
        logger.warning(f"⚠️ Unknown material '{material}', falling back to 'wood'")
        material = "wood"

    # Санитизация пользовательской строки object_name (защита от prompt injection)
    object_name = sanitize_prompt_text(object_name)

    # Валидация параметров инференса
    if guidance_scale < 1.5:
        logger.warning(f"⚠️ guidance_scale={guidance_scale} too low, clamping to 3.5")
        guidance_scale = 3.5

    if _predictor is None or _pipe is None:
        logger.error("❌ Models not loaded")
        raise HTTPException(503, "Models not loaded")

    # Чтение тела запроса (ввод/вывод — асинхронно, не блокирует event loop)
    img_bytes = await image.read()
    logger.info(f"   Image size: {len(img_bytes)} bytes")

    # Проверка размера загружаемого файла (защита от DoS)
    if len(img_bytes) > MAX_IMAGE_SIZE_BYTES:
        logger.warning(
            f"⚠️ Image too large: {len(img_bytes)} bytes "
            f"(limit {MAX_IMAGE_SIZE_BYTES} bytes)"
        )
        raise HTTPException(
            413,
            f"Image too large. Maximum allowed size is "
            f"{MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    logger.info(
        f"   Queued job (concurrency semaphore: "
        f"{_MAX_CONCURRENT_JOBS} max, {_inference_semaphore._value} free)"
    )

    # Тяжёлый инференс (SAM-2 + FLUX) выполняется в отдельном потоке пула,
    # чтобы не блокировать event loop. Семафор ограничивает число одновременных
    # GPU-задач, остальные запросы параллельно ждут своей очереди и могут
    # обрабатываться конкурентно (валидация/декод/отдача на других воркерах).
    async with _inference_semaphore:
        try:
            response_content = await run_in_threadpool(
                run_recolor_job,
                img_bytes,
                point_x,
                point_y,
                material,
                color_hex,
                color_name,
                object_name,
                strength,
                guidance_scale,
                num_inference_steps,
                patina,
                color_r,
                color_g,
                color_b,
            )
        except HTTPException:
            # Пробрасываем корректные HTTP-ошибки (400/413/503 и т.д.) без подмены на 500
            raise
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Request failed after {total_time:.2f}s: {e}")
            logger.error(traceback.format_exc())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise HTTPException(500, str(e))

    total_time = time.time() - start_time
    logger.info(f"✅ Request completed in {total_time:.2f}s total")
    return Response(content=response_content, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    # Запуск с несколькими воркерами (процессами) для реального параллелизма
    # запросов. Число воркеров берётся из переменной окружения WEB_CONCURRENCY
    # (по умолчанию 2). При использовании нескольких воркеров каждый процесс
    # самостоятельно загружает модели в lifespan.
    _workers = int(os.getenv("WEB_CONCURRENCY", "2"))
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=_workers)