import base64
import io
import os
import re
from typing import Optional

import cairosvg
import httpx
import vtracer
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter

app = FastAPI(title="Vectorization API", version="1.2.0")

# Allow the bundled web UI (or any site you embed it on) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration (set these as environment variables on Render) ---
# VECTORIZE_API_KEY: if set, /vectorize requires header "X-API-Key: <value>".
#                     Leave unset while testing; set it before going public
#                     to stop strangers from running up your compute bill.
# MAX_UPLOAD_MB:      hard cap on upload size (default 8MB).
# MAX_DIMENSION:      images are downscaled to at most this many px per side
#                     before tracing, to keep CPU/memory bounded.
API_KEY = os.environ.get("VECTORIZE_API_KEY")
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", 8))
MAX_DIMENSION = int(os.environ.get("MAX_DIMENSION", 1600))

SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/jfif"}

# LLM text-recreation (opt-in): set OPENAI_API_KEY (or LLM_API_KEY) on Render.
# Works with OpenAI, or any OpenAI-compatible endpoint via LLM_BASE_URL
# (e.g. local Ollama: http://localhost:11434/v1 for qwen2-vl / moondream on your RX 5600 XT).
LLM_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


def check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


LLM_PROMPT = (
    "You are an expert vector graphics designer. Recreate the logo in the image as a clean, minimal SVG.\n"
    "Rules:\n"
    "- Use <text> elements for ALL text (preserve exact spelling, accents, numbers). Match font style (sans-serif for BODEG\u00d3N ARGENTINO / 1810, cursive/script for Parrilla criolla) and color (#FFFFFF white on black, #FF8C00 orange for the flame). Center the layout.\n"
    "- For non-text shapes (the double circles, horizontal lines, flame icon) use <circle>/<path>/<line> with the correct strokes.\n"
    "- ViewBox 0 0 500 600. If background is black, add <rect width=\"100%\" height=\"100%\" fill=\"#000\"/> as first element.\n"
    "- Do NOT embed a raster <image> tag.\n"
    "- Return ONLY raw SVG code starting with <svg and ending with </svg>. No markdown fences, no explanation."
)


async def llm_generate_svg(image_bytes: bytes) -> str:
    if not LLM_API_KEY:
        raise HTTPException(status_code=400, detail="LLM not configured: set OPENAI_API_KEY (or LLM_API_KEY) env var. For local RX 5600 XT, run Ollama with qwen2-vl and set LLM_BASE_URL=http://localhost:11434/v1")
    b64 = base64.b64encode(image_bytes).decode()
    # 1.5 MB cap for LLM payload (downscale if needed)
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "temperature": 0.1,
        "max_tokens": 4000,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": LLM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ]},
        ],
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"LLM error {r.status_code}: {r.text[:400]}")
        data = r.json()
        content = data["choices"][0]["message"]["content"]
    # strip markdown fences if model added them
    content = re.sub(r"^```(?:svg)?\s*", "", content.strip(), flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content.strip())
    m = re.search(r"<svg[\s\S]*</svg>", content, re.IGNORECASE)
    if not m:
        raise HTTPException(status_code=502, detail="LLM did not return valid SVG")
    return m.group(0)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/vectorize")
async def vectorize_image(
    file: UploadFile = File(...),
    format: str = Query("svg", pattern="^(svg|pdf)$", description="Output format: svg or pdf"),
    flatten_background: str = Query(
        "none",
        pattern="^(none|white)$",
        description="Flatten a transparent background to white before tracing",
    ),
    blur: float = Query(
        0.0,
        ge=0.0,
        le=0.5,
        description="Gaussian blur radius (0-0.5) to smooth input before tracing.",
    ),
    simplify: float = Query(
        1.0,
        ge=0.0,
        le=3.0,
        description="Curve simplification tolerance (0-3). Higher = smoother, fewer nodes. 0 disables.",
    ),
    preset: str = Query(
        "balanced",
        pattern="^(balanced|smooth|sharp)$",
        description="Preset: balanced (default), smooth (silhouettes, fewer nodes), sharp (small text, more detail)",
    ),
    llm_text: bool = Query(
        False,
        description="If true, use vision LLM to re-typeset text as <text> instead of tracing pixels (needs OPENAI_API_KEY). Opt-in, off by default.",
    ),
    upscale: int = Query(
        1,
        ge=1,
        le=3,
        description="Lanczos upscale factor before tracing (1-3). 2× helps small text; capped by MAX_DIMENSION.",
    ),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    check_api_key(x_api_key)

    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Supported formats: {', '.join(sorted(SUPPORTED_TYPES))}.",
        )

    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Max is {MAX_UPLOAD_MB}MB.",
        )

    # LLM text-recreation branch (bypasses vtracer)
    if llm_text:
        try:
            svg_string = await llm_generate_svg(image_bytes)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM vectorization failed: {e}")

    else:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_format = img.format or "PNG"

            # Optional Lanczos upscale before thumbnail (helps small text, capped by MAX_DIMENSION)
            if upscale > 1:
                w, h = img.size
                # don't upscale beyond ~1.5× MAX_DIMENSION to stay inside free-tier RAM
                max_side = max(w, h) * upscale
                if max_side <= int(MAX_DIMENSION * 1.5):
                    img = img.resize((w * upscale, h * upscale), Image.LANCZOS)

            # Bound memory/CPU during tracing regardless of the source resolution.
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

            # Optional Gaussian blur to smooth gradients and reduce tracing noise.
            # This produces fewer, cleaner SVG paths at the cost of fine detail.
            if blur > 0.0:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur))

            # Logos are frequently PNGs with transparent backgrounds. vtracer
            # traces the alpha channel as a shape, which can produce odd
            # artifacts — flattening to white first gives cleaner results.
            if flatten_background == "white" and img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
                img_format = "PNG"

            buf = io.BytesIO()
            img.save(buf, format=img_format)
            processed_bytes = buf.getvalue()

            # Preset-tuned parameters: smooth keeps silhouettes clean,
            # sharp preserves small text / fine detail, balanced is the middle ground.
            if preset == "smooth":
                p_speckle, p_layer, p_corner, p_length = 10, 32, 70, 5.5
            elif preset == "sharp":
                p_speckle, p_layer, p_corner, p_length = 2, 8, 45, 3.5
            else:  # balanced
                p_speckle, p_layer, p_corner, p_length = 4, 16, 60, 4.0
            # blur slightly lengtens segments for even smoother output
            if blur > 0.0:
                p_length += 1.0

            svg_string = vtracer.Config(
                clustering="color-cluster",
                hierarchical="stacked",
                mode="spline",
                filter_speckle=p_speckle,
                color_precision=6,
                layer_difference=p_layer,
                corner_threshold=p_corner,
                length_threshold=p_length,
                max_iterations=10,
                splice_threshold=45,
                path_precision=3,
                simplify=simplify if simplify > 0 else None,
            ).convert_bytes(processed_bytes)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vectorization failed: {e}")

    if format == "svg":
        return StreamingResponse(
            io.BytesIO(svg_string.encode("utf-8")),
            media_type="image/svg+xml",
            headers={"Content-Disposition": "attachment; filename=vector.svg"},
        )

    try:
        pdf_bytes = cairosvg.svg2pdf(bytestring=svg_string.encode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {e}")
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=vector.pdf"},
    )


# Serves static/index.html at "/" — a simple upload UI. Registered last so
# it doesn't shadow the API routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
