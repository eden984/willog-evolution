"""Render the Willog evolution motion graphic.

usage:
  python render.py frames OUT_DIR t1 t2 ...   # still frames (PNG) at given seconds
  python render.py video                      # full MP4 with bgm.wav muxed in
"""
import subprocess, sys, time
from pathlib import Path
import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "index.html"
FPS, DURATION = 30, 30.0


def open_page(p):
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    page.goto(SRC.as_uri())
    page.evaluate("window.ready")
    page.wait_for_function("anims.length > 0")
    # make sure the web fonts are really in before capturing (window.ready may time out on slow networks)
    page.wait_for_function("window.gfReady && document.fonts.check('900 40px \"Noto Sans KR\"')", timeout=15000)
    page.evaluate("playing=false; document.getElementById('start').style.display='none';"
                  "document.getElementById('ui').style.display='none'")
    return browser, page


def seek(page, sec):
    page.evaluate(f"t = {sec}; render()")


def frames(out_dir, times):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser, page = open_page(p)
        for s in times:
            seek(page, float(s))
            page.screenshot(path=str(out / f"f_{float(s):05.2f}.png"))
        browser.close()


def video():
    audio = ROOT / "bgm.wav"
    out = ROOT / "willog-evolution.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-"]
    if audio.exists():
        cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", "-movflags", "+faststart", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(FPS * DURATION) + 1
    with sync_playwright() as p:
        browser, page = open_page(p)
        start = time.time()
        for i in range(n):
            seek(page, i / FPS)
            ff.stdin.write(page.screenshot(type="png"))
            if i % 150 == 0:
                print(f"frame {i}/{n - 1} ({time.time() - start:.0f}s)", flush=True)
        browser.close()
    ff.stdin.close()
    sys.exit(ff.wait())


if __name__ == "__main__":
    if sys.argv[1] == "frames":
        frames(sys.argv[2], sys.argv[3:])
    else:
        video()
