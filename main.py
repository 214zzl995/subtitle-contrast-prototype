from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("subtitle_contrast_prototype.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
