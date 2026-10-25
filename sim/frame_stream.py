"""Camera output for sim rollouts: async H.264 recording and a live MJPEG stream (works headless).

Both run off the sim thread. `push()` only stores a reference / enqueues; encoding happens in worker
threads, so the sim loop never waits for ffmpeg or JPEG. The live stream encodes only while a browser
is connected: open http://<host>:<port>/ to watch.
"""

import http.server
import os
import queue
import socketserver
import threading

import cv2
import numpy as np

HUD_HEIGHT = 28
HUD_LINES = 4


def compose(images: dict, keys, hud: str = "", scale: int = 1) -> np.ndarray:
    """Tile the camera images side by side (RGB uint8) with a status bar underneath, one row per hud line."""
    tiles = np.concatenate([images[k] for k in keys], axis=1)
    if scale != 1:
        tiles = cv2.resize(tiles, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # Fixed bar height: a video stream cannot change frame size mid-file.
    lines = (hud.split("\n") if hud else [])[:HUD_LINES]
    bar = np.full((HUD_HEIGHT * HUD_LINES, tiles.shape[1], 3), 24, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (8, 19 + i * HUD_HEIGHT), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1,
                    cv2.LINE_AA)
    return np.concatenate([tiles, bar], axis=0)


class VideoRecorder:
    """H.264 mp4 via imageio-ffmpeg in a worker thread; frames are written in order, none dropped.

    With `frames_dir`, every frame is also saved as frames_dir/NNNNNN.jpg, so a video can be rebuilt even if
    the process dies before the mp4 is finalised:  ffmpeg -framerate FPS -i frames_dir/%06d.jpg out.mp4
    """

    def __init__(self, path: str, fps: float = 30, crf: int = 20, preset: str = "ultrafast",
                 frames_dir: str | None = None):
        self.path, self.fps, self.crf, self.preset, self.frames_dir = path, fps, crf, preset, frames_dir
        if frames_dir:
            os.makedirs(frames_dir, exist_ok=True)
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.frames = 0
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="video-recorder", daemon=True)
        self.thread.start()

    def push(self, frame: np.ndarray) -> None:
        if self.error is None:
            self.q.put(frame)  # blocks only if the encoder is >256 frames behind

    def _run(self) -> None:
        import imageio_ffmpeg

        writer = None
        try:
            while (frame := self.q.get()) is not None:
                if self.frames_dir:
                    cv2.imwrite(os.path.join(self.frames_dir, f"{self.frames:06d}.jpg"),
                                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = imageio_ffmpeg.write_frames(
                        self.path, (w, h), fps=self.fps, codec="libx264", pix_fmt_out="yuv420p", quality=None,
                        macro_block_size=1, output_params=["-preset", self.preset, "-crf", str(self.crf)],
                    )
                    writer.send(None)
                writer.send(np.ascontiguousarray(frame))
                self.frames += 1
        except Exception as e:  # keep the rollout alive; report at close
            self.error = e
            while self.q.get() is not None:
                pass
        finally:
            if writer is not None:
                writer.close()

    def close(self) -> None:
        self.q.put(None)
        self.thread.join()
        if self.error is not None:
            print(f"[video] FAILED after {self.frames} frames: {self.error!r}", flush=True)


_PAGE = b"""<!doctype html><html><head><title>FR3 Duo rollout</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif;text-align:center}
img{max-width:100%;image-rendering:pixelated;margin-top:8px}</style></head>
<body><div>FR3 Duo closed-loop rollout: external | left wrist | right wrist</div><img src="/stream"></body></html>"""


class LiveStream:
    """MJPEG over HTTP. Keeps only the latest frame; each connected client gets every new frame it can take."""

    def __init__(self, port: int, quality: int = 80):
        self.port, self.quality = port, quality
        self.cond = threading.Condition()
        self.frame: np.ndarray | None = None
        self.seq = 0
        self.clients = 0
        self.closed = False
        stream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path != "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(_PAGE)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with stream.cond:
                    stream.clients += 1
                seen = -1
                try:
                    while True:
                        with stream.cond:
                            stream.cond.wait_for(lambda: stream.closed or stream.seq != seen, timeout=5.0)
                            if stream.closed:
                                return
                            frame, seen = stream.frame, stream.seq
                        if frame is None:
                            continue
                        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, stream.quality])
                        if not ok:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with stream.cond:
                        stream.clients -= 1

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.server = Server(("0.0.0.0", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="live-stream", daemon=True)
        self.thread.start()
        print(f"[stream] live view at http://localhost:{port}/", flush=True)

    @property
    def watching(self) -> bool:
        return self.clients > 0

    def push(self, frame: np.ndarray) -> None:
        with self.cond:
            self.frame = frame
            self.seq += 1
            self.cond.notify_all()

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.server.shutdown()
        self.server.server_close()
