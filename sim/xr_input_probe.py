"""Print what Isaac Sim receives from the Quest controllers, to see where controller input gets lost.

Run in the Script Editor while the VR session is on, then move both thumbsticks and press the triggers
for 15 s. Output goes to the Script Editor console and the Kit log.

    exec(open("<repo>/sim/xr_input_probe.py").read())
"""

import asyncio
import time

import omni.kit.app
from omni.kit.xr.core import XRCore

DURATION_S = 15.0
PERIOD_S = 0.5


async def _probe():
    app = omni.kit.app.get_app()
    core = XRCore.get_singleton()
    print(f"[probe] xr viewport enabled: {core.is_xr_viewport_enabled()}, "
          f"profile: {core.get_current_profile_name()}", flush=True)

    devices = core.get_all_input_devices()
    print(f"[probe] input devices: {[str(d.get_name()) for d in devices]}", flush=True)
    for d in devices:
        inputs = {str(i): [str(g) for g in d.get_input_gesture_names(i)] for i in d.get_input_names()}
        print(f"[probe] {d.get_name()} inputs: {inputs}", flush=True)

    end = time.time() + DURATION_S
    next_print = 0.0
    while time.time() < end:
        await app.next_update_async()
        if time.time() < next_print:
            continue
        next_print = time.time() + PERIOD_S
        parts = []
        for d in core.get_all_input_devices():
            name = str(d.get_name())
            if "hand" not in name:
                continue
            vals = {}
            for inp in d.get_input_names():
                for g in d.get_input_gesture_names(inp):
                    v = d.get_input_gesture_value(inp, g)
                    if abs(v) > 0.05:
                        vals[f"{inp}.{g}"] = round(v, 2)
            pos = d.get_pose().ExtractTranslation()
            parts.append(f"{name} pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) active={vals}")
        print("[probe] " + " | ".join(parts), flush=True)
    print("[probe] done", flush=True)


async def _probe_safe():
    try:
        await _probe()
    except Exception:
        import traceback
        print("[probe] FAILED:\n" + traceback.format_exc(), flush=True)


asyncio.ensure_future(_probe_safe())
