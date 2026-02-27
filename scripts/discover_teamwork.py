"""
Quick discovery script to list Teamwork workflows and stages (board columns)
for the allowed projects.

Usage:
  cd BA_Agent && python scripts/discover_teamwork.py
"""

import asyncio
import base64
import json
import os
import sys

# Add parent to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TEAMWORK_API_KEY", "")
DOMAIN = os.getenv("TEAMWORK_DOMAIN", "")
PROJECT_IDS = ["820301", "822891"]

if not API_KEY or not DOMAIN:
    print("ERROR: TEAMWORK_API_KEY and TEAMWORK_DOMAIN must be set in .env")
    sys.exit(1)


def headers():
    encoded = base64.b64encode(f"{API_KEY}:x".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


BASE_V3 = f"https://{DOMAIN}/projects/api/v3"
BASE_V1 = f"https://{DOMAIN}"


async def main():
    async with httpx.AsyncClient(headers=headers(), timeout=30) as client:
        for pid in PROJECT_IDS:
            print(f"\n{'='*60}")
            print(f"PROJECT {pid}")
            print(f"{'='*60}")

            # Get project name
            r = await client.get(f"{BASE_V3}/projects/{pid}.json")
            if r.is_success:
                proj = r.json().get("project", {})
                print(f"  Name: {proj.get('name')}")
            else:
                print(f"  FAILED to fetch project: {r.status_code}")
                continue

            # --- V3 Workflows ---
            print(f"\n  --- V3 Workflows ---")
            r = await client.get(f"{BASE_V3}/projects/{pid}/workflows.json")
            if r.is_success:
                data = r.json()
                workflows = data.get("workflows", [])
                if not workflows:
                    print("  No workflows found for this project.")
                for wf in workflows:
                    wf_id = wf.get("id")
                    print(f"\n  Workflow: {wf.get('name')} (id={wf_id})")
                    print(f"    projectSpecific: {wf.get('projectSpecific')}")

                    # Get stages for this workflow
                    sr = await client.get(f"{BASE_V3}/workflows/{wf_id}/stages.json")
                    if sr.is_success:
                        stages = sr.json().get("stages", [])
                        for s in stages:
                            print(f"    Stage: {s.get('name'):<30} id={s.get('id'):<10} color={s.get('color')} displayOrder={s.get('displayOrder')}")
                    else:
                        print(f"    FAILED to fetch stages: {sr.status_code} {sr.text[:200]}")
            else:
                print(f"  FAILED to fetch workflows: {r.status_code} {r.text[:200]}")

            # --- V1 Board Columns (legacy) ---
            print(f"\n  --- V1 Board Columns (legacy) ---")
            r = await client.get(f"{BASE_V1}/projects/{pid}/boards/columns.json")
            if r.is_success:
                data = r.json()
                columns = data.get("columns", [])
                if not columns:
                    print("  No board columns found.")
                for col in columns:
                    print(f"    Column: {col.get('name'):<30} id={col.get('id'):<10} color={col.get('color')}")
            else:
                print(f"  FAILED to fetch board columns: {r.status_code}")

            # --- Task lists ---
            print(f"\n  --- Task Lists ---")
            r = await client.get(f"{BASE_V3}/projects/{pid}/tasklists.json")
            if r.is_success:
                for tl in r.json().get("tasklists", []):
                    print(f"    TaskList: {tl.get('name'):<30} id={tl.get('id')}")
            else:
                print(f"  FAILED to fetch task lists: {r.status_code}")

        # --- Also test file upload presigned URL (dry run) ---
        print(f"\n{'='*60}")
        print("FILE UPLOAD TEST (presigned URL)")
        print(f"{'='*60}")
        r = await client.get(
            f"{BASE_V1}/projects/api/v1/pendingfiles/presignedurl.json",
            params={"fileName": "test-spec.md", "fileSize": 100},
        )
        if r.is_success:
            data = r.json()
            print(f"  ref: {data.get('ref')}")
            print(f"  url: {data.get('url', '')[:100]}...")
            print("  File upload presigned URL endpoint works!")
        else:
            print(f"  FAILED: {r.status_code} {r.text[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
