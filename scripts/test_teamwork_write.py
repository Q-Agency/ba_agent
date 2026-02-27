"""
Quick test: upload a dummy SPEC.md to a Teamwork task and move it to Ready for Design.

Usage:
  cd BA_Agent && uv run python scripts/test_teamwork_write.py <TASK_ID>

Example:
  uv run python scripts/test_teamwork_write.py 12345678
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

load_dotenv()

from app.routers.teamwork import upload_spec_to_task, move_task_to_stage


DUMMY_SPEC = """\
# SPEC: Test Task

## Overview
This is a test SPEC uploaded via the BA Agent API.

## User Stories
- As a developer, I can verify file uploads work.

## Acceptance Criteria
- [x] File appears as attachment on Teamwork task
- [x] Task moves to "Ready for Design" column
"""


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/test_teamwork_write.py <TASK_ID>")
        print("\nProvide a Teamwork task ID to test against.")
        sys.exit(1)

    task_id = sys.argv[1]
    print(f"Testing with task ID: {task_id}")

    # 1. Upload SPEC
    print("\n--- Uploading SPEC.md ---")
    try:
        ref = await upload_spec_to_task(task_id, DUMMY_SPEC, "Test Task")
        print(f"  SUCCESS: ref={ref}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 2. Move to Ready for Design
    print("\n--- Moving to Ready for Design ---")
    try:
        await move_task_to_stage(task_id)
        print("  SUCCESS: task moved to Ready for Design")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\nDone! Check the task in Teamwork to verify.")


if __name__ == "__main__":
    asyncio.run(main())
