# Vectorization microservice

Converts JPG/PNG/WEBP/BMP images to vector SVG or PDF using vtracer (no AI,
no per-image API cost — just server compute).

## Deploy to Render

1. Push this folder to a new GitHub repo.
2. On Render: **New → Web Service**, connect the repo.
3. Environment: **Docker** (Render will detect the Dockerfile automatically).
4. Set environment variables (Render dashboard → Environment):
   - `VECTORIZE_API_KEY` — leave unset to test, set a secret value before
     making the URL public so strangers can't run up your compute bill.
   - `MAX_UPLOAD_MB` — default 8.
   - `MAX_DIMENSION` — default 1600 (higher = sharper traces, slower/costlier).
5. Plan:
   - **Free** — $0/mo, but spins down after ~15 min idle; the next request
     takes 30-50s to wake up. Fine for internal/occasional use.
   - **Starter** (~$7/mo) — always warm, no cold starts. Worth it if this
     is customer-facing.
6. Deploy. Your service will be at `https://<your-service>.onrender.com`.

## Using it

- Open the root URL in a browser for the built-in upload page.
- Or POST directly:
  ```
  curl -X POST "https://<your-service>.onrender.com/vectorize?format=svg" \
    -H "X-API-Key: <your key, if set>" \
    -F "file=@logo.png"
  ```

## Optional: expose it as a ChatGPT Custom GPT

If you'd rather people vectorize images by chatting with a GPT instead of
using the web page:

1. In ChatGPT, go to **Explore GPTs → Create**.
2. Under **Configure → Actions**, paste the contents of `openapi.json`
   (after replacing `YOUR-RENDER-URL` with your real Render URL).
3. Publish the GPT.

This route needs a **ChatGPT Plus/Team account**, not an OpenAI API key —
the API key is only needed if you're calling OpenAI's models directly from
your own code, which this project doesn't do.
