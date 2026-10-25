"""Open a USD stage in the Isaac Sim GUI (pass via --exec; --/app/file/open is ignored).

Usage: USD=/path/to/scene.usd isaacsim isaacsim.exp.full --exec sim/open_scene.py
"""
import asyncio
import os

import omni.usd

USD = os.environ.get("USD", "Hackathon_1stDay.usd")


async def _open():
    ok, err = await omni.usd.get_context().open_stage_async(USD)
    print(f"[open_scene] {USD}: {'opened' if ok else 'FAILED ' + str(err)}")


asyncio.ensure_future(_open())
