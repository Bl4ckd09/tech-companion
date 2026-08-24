from __future__ import annotations

import argparse
import asyncio
import json
import tarfile
from pathlib import Path

import httpx

from app.pioneer.client import PioneerClient
from pioneer_pipeline_state import REPO_ROOT, load_api_key

OUTPUT_ROOT = REPO_ROOT / ".state" / "pioneer_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract one completed Pioneer model."
    )
    parser.add_argument("job_id")
    return parser.parse_args()


async def run(job_id: str) -> None:
    client = PioneerClient(load_api_key(), timeout_seconds=60)
    try:
        metadata = await client.model_download(job_id)
    finally:
        await client.close()

    download_url = metadata.get("download_url")
    filename = metadata.get("file_name")
    if not isinstance(download_url, str) or not download_url:
        raise RuntimeError("Pioneer did not return a model download URL")
    if not isinstance(filename, str) or not filename.endswith(".tar.gz"):
        raise RuntimeError("Pioneer returned an unexpected model archive name")

    output_dir = OUTPUT_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / filename
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as downloader:
        response = await downloader.get(download_url)
        response.raise_for_status()
    archive_path.write_bytes(response.content)

    extracted_dir = output_dir / "weights"
    extracted_dir.mkdir(exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted_dir, filter="data")

    result = {
        "job_id": job_id,
        "archive": str(archive_path.relative_to(REPO_ROOT)),
        "weights": str(extracted_dir.relative_to(REPO_ROOT)),
        "files": sorted(
            str(path.relative_to(extracted_dir))
            for path in extracted_dir.rglob("*")
            if path.is_file()
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(run(parse_args().job_id))
