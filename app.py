import os
import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# تنظیمات لاگینگ استاندارد جهت رصد عملکرد در HuggingFace Spaces
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("fno_inference_service")


# ==========================================
# ۱. کلاس تنظیمات و پیکربندی معماری FNO
# ==========================================
@dataclass(frozen=True)
class FNOConfig:
    """
    تنظیمات غیرقابل تغییر معماری مدل FNO هماهنگ با مشخصات آموزش‌شده.
    """
    in_channels: int = 2          # ورودی: میدان‌های سرعت و دما (۲ کانال مختصات خودکار توسط کتابخانه اضافه می‌شود)
    out_channels: int = 3         # خروجی: [Velocity_X, Velocity_Y, Temperature]
    n_modes: Tuple[int, int] = (16, 16)
    hidden_channels: int = 64
    factorization: None = None
    norm: None = None
    device: str = "cpu"           # استقرار بهینه روی CPU-Only Spaces
    weights_path_npz: str = "weights_real_v2.npz"
    weights_path_pth: str = "weights_real_v2.pth"  # مسیر جایگزین (Fallback)
    
    # ثابت‌های ریاضی پیش‌پردازش (Forward Min-Max Scaling Constants)
    X_MIN: Tuple[float, float] = (0.0, 0.0)
    X_MAX: Tuple[float, float] = (2.997745990753174, 303.1484069824219)
    
    # ثابت‌های ریاضی پس‌پردازش (Inverse Min-Max Scaling Constants - آماده برای Broadcasting)
    Y_MIN: torch.Tensor = torch.tensor(
        [-0.8491503000259399, -3.465104103088379, 288.1639404296875],
        dtype=torch.float32,
        device="cpu"
    ).view(1, 3, 1, 1)
    Y_MAX: torch.Tensor = torch.tensor(
        [3.000093936920166, 1.088096022605896, 310.1499938964844],
        dtype=torch.float32,
        device="cpu"
    ).view(1, 3, 1, 1)


# ==========================================
# ۲. تابع بارگذاری ایمن اوزان (Zero Cold-Start)
# ==========================================
def load_fno_model(config: FNOConfig) -> torch.nn.Module:
    """
    نمونه‌سازی از مدل FNO و بارگذاری ایمن اوزان از فایل .npz (یا .pth)
    با تبدیل دقیق آرایه‌های NumPy و مدیریت حافظه C-Contiguous.
    """
    logger.info("Initializing FNO model architecture on %s...", config.device)
    
    try:
        from neuralop.models import FNO
    except ImportError as exc:
        logger.critical("Library 'neuraloperator' is not installed or incompatible.")
        raise RuntimeError("neuraloperator package missing.") from exc

    # ۱. ایجاد نمونه از مدل
    model = FNO(
        n_modes=config.n_modes,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_channels=config.hidden_channels,
        factorization=config.factorization,
        norm=config.norm
    ).to(config.device)

    # ۲. بررسی و بارگذاری اوزان
    npz_path = config.weights_path_npz
    pth_path = config.weights_path_pth

    if os.path.exists(npz_path):
        logger.info("Loading weights safely from compressed NumPy archive: '%s'", npz_path)
        try:
            # بارگذاری کاملاً ایمن بدون وابستگی به Pickle
            with np.load(npz_path, allow_pickle=False) as npz_file:
                state_dict: Dict[str, Any] = {}
                for key in npz_file.files:
                    arr = npz_file[key]
                    
                    # تضمین چیدمان پیوسته حافظه (C-Contiguous) جهت جلوگیری از خطا در PyTorch
                    if not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    
                    # تبدیل به تانسور PyTorch (پشتیبانی خودکار از complex64 برای مدهای فرکانسی)
                    tensor = torch.from_numpy(arr)
                    state_dict[key] = tensor

            # تزریق اوزان به مدل
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning("Missing keys during NPZ loading: %s", missing_keys)
            if unexpected_keys:
                logger.warning("Unexpected keys during NPZ loading: %s", unexpected_keys)
                
            logger.info("Successfully loaded weights from NPZ.")
            
        except Exception as exc:
            logger.error("Failed to load NPZ file: %s. Attempting fallback...", exc)
            _fallback_load_pth(model, pth_path, config.device)
            
    elif os.path.exists(pth_path):
        logger.info("NPZ file not found. Loading from fallback PyTorch file: '%s'", pth_path)
        _fallback_load_pth(model, pth_path, config.device)
    else:
        raise FileNotFoundError(
            f"No valid weights file found at '{npz_path}' or '{pth_path}'."
        )

    # ۳. تنظیم مدل به حالت استنتاج جهت غیرفعال‌سازی گرادیان‌ها و رفتار بهینه
    model.eval()
    logger.info("Model evaluation mode enabled. Inference readiness: 100%%.")
    return model


def _fallback_load_pth(model: torch.nn.Module, pth_path: str, device: str) -> None:
    """بارگذاری کمکی از فایل .pth در صورت عدم دسترسی به .npz"""
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Fallback weight file '{pth_path}' does not exist.")
    
    # استفاده از weights_only=True برای امنیت حداکثری
    state_dict = torch.load(pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    logger.info("Successfully loaded weights from PTH fallback.")

# ==========================================
# ۲.۵. اسکیمای Pydantic و پیش‌پردازش ورودی‌ها
# ==========================================
class SimulationRequest(BaseModel):
    """
    اسکیمای اعتبارسنجی ورودی‌های کلاینت بر اساس بازه فیزیکی مجاز.
    """
    velocity: float = Field(
        ...,
        ge=0.5,
        le=3.0,
        description="Inlet velocity in m/s (valid physical range: 0.5 - 3.0)",
        json_schema_extra={"example": 1.5}
    )
    temperature: float = Field(
        ...,
        ge=288.15,
        le=303.15,
        description="Inlet temperature in Kelvin (valid physical range: 288.15 - 303.15)",
        json_schema_extra={"example": 295.15}
    )


def prepare_input_tensor(
    request: SimulationRequest,
    x_min: Tuple[float, float],
    x_max: Tuple[float, float],
    device: str = "cpu"
) -> torch.Tensor:
    """
    پیش‌پردازش ریاضی ورودی: مقیاس‌دهی Forward Min-Max و تزریق شرط مرزی (Spatial Masking)
    به صورت کاملاً بردارسازی‌شده و بدون حلقه‌های پایتونی.
    
    ابعاد تانسور خروجی: (1, 2, 64, 64)
    """
    # ۱. مقیاس‌دهی خطی Min-Max
    vel_scaled = (request.velocity - x_min[0]) / (x_max[0] - x_min[0])
    temp_scaled = (request.temperature - x_min[1]) / (x_max[1] - x_min[1])

    # ۲. ساخت تانسور صفر اولیه با ابعاد (1, 2, 64, 64) روی دستگاه هدف (cpu)
    input_tensor = torch.zeros((1, 2, 64, 64), dtype=torch.float32, device=device)

    # ۳. ماسک‌گذاری فضایی شرط مرزی (Trombe Wall Inlet Slice) با عملیات برشی سریع
    input_tensor[0, 0, 50:58, 0] = vel_scaled
    input_tensor[0, 1, 50:58, 0] = temp_scaled

    return input_tensor

# ==========================================
# ۲.۶. موتور استنتاج سریع و مقیاس‌دهی معکوس
# ==========================================
@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    y_min: torch.Tensor,
    y_max: torch.Tensor
) -> torch.Tensor:
    """
    اجرای استنتاج سریع مدل FNO روی CPU و اعمال مقیاس‌دهی معکوس (Inverse Min-Max Scaling)
    جهت تبدیل خروجی خام شبکه به میدان‌های فیزیکی واقعی.

    تضمین‌های کارایی:
    - استفاده از دکوراتور @torch.inference_mode جهت غیرفعال‌سازی کامل گراف گرادیان (Zero autograd overhead).
    - اعمال عملیات ریاضی بردارسازی‌شده (Broadcasted over 1x3x1x1) بدون حلقه for.
    - جلوگیری از نشت حافظه و کپی‌های اضافی در RAM.

    ابعاد ورودی: (1, 2, 64, 64)
    ابعاد خروجی: (1, 3, 64, 64) -> [Velocity_X, Velocity_Y, Temperature]
    """
    # ۱. استنتاج خام مدل (Forward Pass)
    raw_output = model(input_tensor)

    # ۲. اطمینان از هم‌خوانی دستگاه و نوع داده تانسورهای ثابت با خروجی
    # (در صورتی که روی همان device و dtype باشند، هیچ کپی اضافی در RAM ایجاد نمی‌شود)
    y_min_t = y_min.to(device=raw_output.device, dtype=raw_output.dtype)
    y_max_t = y_max.to(device=raw_output.device, dtype=raw_output.dtype)

    # ۳. مقیاس‌دهی معکوس بردارسازی‌شده (Broadcasted Inverse Min-Max Transform)
    # فرمول: output_physical = raw_output * (Y_MAX - Y_MIN) + Y_MIN
    output_physical = raw_output * (y_max_t - y_min_t) + y_min_t

    return output_physical
# ==========================================
# ۳. مدیریت طول عمر وب‌سرویس (Lifespan Manager)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    مدیریت استارت‌آپ و شات‌داون کانتینر در HuggingFace Spaces.
    لود مدل در زمان بوت، زمان Cold-Start درخواست‌های POST را به 0ms می‌رساند.
    """
    logger.info("=== Starting up SciML/CFD FNO Inference Service ===")
    config = FNOConfig()
    
    try:
        # بارگذاری یک‌باره مدل روی RAM سیستم در App State
        app.state.config = config
        app.state.model = load_fno_model(config)
        app.state.model_ready = True
        logger.info("=== Service is READY to accept real-time inference requests ===")
    except Exception as exc:
        logger.critical("Fatal error during startup model initialization: %s", exc)
        app.state.model = None
        app.state.model_ready = False
        
    yield  # ورود به فاز پاسخ‌گویی وب‌سرویس
    
    # آزادسازی حافظه در زمان خاموش شدن سرویس
    logger.info("=== Shutting down service. Releasing RAM resources ===")
    app.state.model = None
    app.state.model_ready = False


# ==========================================
# ۴. ساخت اپلیکیشن FastAPI و تنظیمات CORS
# ==========================================
app = FastAPI(
    title="SciML / CFD Real-Time FNO Surrogate API",
    description="Real-time fluid dynamics surrogate inference service using Fourier Neural Operators.",
    version="1.0.0",
    lifespan=lifespan
)

# فعال‌سازی CORS جهت ارتباط بدون مانع با کلاینت‌ها و مرورگرها
# فعال‌سازی CORS جهت ارتباط بدون مانع با کلاینت‌ها و مرورگرها
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://saeidbahrami.com",
        "https://www.saeidbahrami.com",
        "http://saeidbahrami.com",
    ],  # استفاده از لیست صریح دامنه‌ها جهت جلوگیری از تداخل Starlette
    allow_credentials=False,  # سازگار با معماری Cross-Origin بدون کوکی
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # اجازه دسترسی مرورگر به تمامی هدرهای پاسخ سرور
    max_age=3600,  # کش کردن پاسخ OPTIONS (Preflight) به مدت ۱ ساعت جهت کاهش تاخیر
)


# ==========================================
# ۵. مسیر بررسی سلامت سرویس (Health Check)
# ==========================================
@app.get("/", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    بررسی وضعیت زنده بودن وب‌سرویس و صحت استقرار مدل FNO در حافظه RAM.
    """
    model_ready = getattr(app.state, "model_ready", False)
    
    if not model_ready or app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service degraded: FNO Model is not loaded in RAM."
        )

    config: FNOConfig = app.state.config
    return {
        "status": "healthy",
        "service": "SciML/CFD FNO Real-time Surrogate",
        "model_status": {
            "loaded_in_ram": True,
            "device": config.device,
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "n_modes": list(config.n_modes),
            "hidden_channels": config.hidden_channels
        },
        "cold_start_latency": "0ms (Pre-loaded via Lifespan)"
    }

# ==========================================
# ۶. مسیر اصلی استنتاج بلادرنگ (POST /predict)
# ==========================================
import time
from fastapi import Request

@app.post("/predict", status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict(payload: SimulationRequest, request: Request) -> Dict[str, Any]:
    """
    دریافت سرعت و دمای ورودی، اجرای پایپ‌لاین استنتاج بلادرنگ FNO و بازگرداندن
    میدان‌های فیزیکی خروجی در قالب JSON سازگار با جاوااسکریپت.
    """
    # ۱. بررسی سلامت و آماده‌به‌کار بودن مدل در RAM
    if not getattr(request.app.state, "model_ready", False) or request.app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service degraded: FNO Model is not loaded in RAM."
        )

    # شروع سنجش زمان اجرای درخواست (با دقت بالا)
    start_time = time.perf_counter()

    try:
        # ۲. بازیابی تنظیمات و نمونه مدل از State اپلیکیشن
        config: FNOConfig = request.app.state.config
        model: torch.nn.Module = request.app.state.model

        # ۳. پیش‌پردازش ورودی و ساخت تانسور (1, 2, 64, 64)
        input_tensor = prepare_input_tensor(
            request=payload,
            x_min=config.X_MIN,
            x_max=config.X_MAX,
            device=config.device
        )

        # ۴. اجرای استنتاج سریع و اعمال مقیاس‌دهی معکوس فیزیکی -> خروجی (1, 3, 64, 64)
        output_physical = run_inference(
            model=model,
            input_tensor=input_tensor,
            y_min=config.Y_MIN,
            y_max=config.Y_MAX
        )

        # ۵. سریال‌سازی سریع جاوااسکریپت-دوست (Fast JS-Friendly Serialization)
        # حذف بعد بچ (Batch Dimension) جهت تبدیل به ابعاد [3, 64, 64]
        output_squeezed = output_physical.squeeze(0)
        
        # تبدیل سریع به NumPy و سپس tolist() جهت بهینه‌سازی تاخیر سریال‌سازی JSON
        fields_list = output_squeezed.cpu().numpy().tolist()

        # محاسبه زمان کل اجرا بر حسب میلی‌ثانیه
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "Inference request processed successfully in %.2f ms (v=%.2f, T=%.2f).",
            execution_time_ms,
            payload.velocity,
            payload.temperature
        )

        # ۶. بازگرداندن پاسخ نهایی با ساختار استاندارد
        return {
            "status": "success",
            "fields": fields_list,
            "channels": ["Velocity_X", "Velocity_Y", "Temperature"],
            "execution_time_ms": round(execution_time_ms, 2)
        }

    except HTTPException:
        # ارسال مجدد خطاهای HTTP مدیریت‌شده بدون تغییر
        raise
    except Exception as exc:
        # مدیریت خطا و ثبت لاگ کامل در صورت بروز هرگونه مشکل محاسباتی
        logger.error("Computation failed during FNO inference pipeline: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Computation failed during FNO inference: {str(exc)}"
        )
if __name__ == "__main__":
    import uvicorn
    # دریافت داینامیک پورت تخصیص‌یافته توسط Render (پیش‌فرض 10000 هماهنگ با معماری Render)
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)