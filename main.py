import io
import os
from typing import Optional

import cairosvg
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


def check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


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

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img_format = img.format or "PNG"

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

        if format == "svg":
            return StreamingResponse(
                io.BytesIO(svg_string.encode("utf-8")),
                media_type="image/svg+xml",
                headers={"Content-Disposition": "attachment; filename=vector.svg"},
            )

        pdf_bytes = cairosvg.svg2pdf(bytestring=svg_string.encode("utf-8"))
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=vector.pdf"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {e}")


# Serves static/index.html at "/" — a simple upload UI. Registered last so
# it doesn't shadow the API routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
