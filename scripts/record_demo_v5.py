"""WorldKit — Apple-style product film with glassmorphism."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT_DIR = Path(__file__).parent.parent / "demo_frames_v5"
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir()

W, H = 1920, 1080
FPS = 30
FI = 0  # frame index

# ── Palette — warm olive / beige / chrome ─────────────
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK = (55, 48, 40)          # warm charcoal-brown
TEXT = (72, 65, 55)           # warm body text
SECONDARY = (148, 138, 122)  # warm mid-gray
TERTIARY = (205, 198, 186)   # warm light
ACCENT = (108, 122, 74)      # olive green
PURPLE = (145, 130, 100)     # warm bronze/chrome
TEAL = (82, 140, 128)        # sage teal
GREEN = (110, 155, 85)       # natural green
ORANGE = (198, 145, 62)      # warm gold
RED = (185, 90, 70)          # muted terracotta

# Gradient BG colors — warm cream/beige
BG_TOP = (250, 247, 240)
BG_BOT = (240, 234, 222)

# Glass — warmer tint
GLASS_FILL = (255, 253, 248, 170)
GLASS_BORDER = (255, 252, 245, 210)

# Data
DATA_PATH = Path(__file__).parent.parent / "data" / "pusht_real.h5"
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).parent.parent / "data" / "pusht_train.h5"
with h5py.File(DATA_PATH, "r") as f:
    PX = np.array(f["pixels"])

# ── Fonts — Outfit (clean geometric sans) ─────────────
FONT_DIR = Path(__file__).parent.parent / "assets" / "fonts"
_OUTFIT = str(FONT_DIR / "Outfit-Variable.ttf")

def _outfit(sz, weight=400):
    """Outfit variable font at a given weight."""
    try:
        f = ImageFont.truetype(_OUTFIT, sz)
        f.set_variation_by_axes([weight])
        return f
    except Exception:
        return ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", sz)

def sf(sz):
    return _outfit(sz, 600)       # SemiBold — section headers

def sf_bold(sz):
    return _outfit(sz, 700)       # Bold — hero titles

def sf_light(sz):
    return _outfit(sz, 300)       # Light — subtitles

def sf_med(sz):
    return _outfit(sz, 500)       # Medium — body

def sf_reg(sz):
    return _outfit(sz, 400)       # Regular — labels

def mono(sz):
    for p in ["/System/Library/Fonts/SFMono-Regular.otf", "/System/Library/Fonts/Menlo.ttc"]:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


# ── Core helpers ───────────────────────────────────────
def ease(t):
    return 1.0 - (1.0 - min(1.0, max(0.0, t))) ** 3

def eio(t):
    t = min(1.0, max(0.0, t))
    return 3*t*t - 2*t*t*t

def L(a, b, t):
    return a + (b - a) * t

def LC(c1, c2, t):
    return tuple(int(L(a, b, t)) for a, b in zip(c1, c2))

def tw(d, s, f):
    b = d.textbbox((0, 0), s, font=f)
    return b[2] - b[0]

def dc(d, y, s, f, c):
    d.text(((W - tw(d, s, f)) // 2, y), s, fill=c, font=f)

def dca(d, x, y, s, f, c):
    d.text((x - tw(d, s, f) // 2, y), s, fill=c, font=f)

def pt(ep, t, sz=None):
    im = Image.fromarray(PX[ep % 200, t % 16])
    return im.resize(sz, Image.LANCZOS) if sz else im

def save(img):
    global FI
    img.save(OUT_DIR / f"frame_{FI:05d}.png")
    FI += 1

def hold(img, n):
    for _ in range(n):
        save(img)

def fout(img, n=12):
    bg = make_bg()
    for i in range(n):
        save(Image.blend(img, bg, eio(i / n)))


# ── Background with subtle gradient + ambient blobs ───
def make_bg():
    """Create background with subtle gradient and ambient color blobs."""
    img = Image.new("RGBA", (W, H), (*BG_TOP, 255))
    d = ImageDraw.Draw(img)
    # Vertical gradient
    for y in range(H):
        t = y / H
        r = int(L(BG_TOP[0], BG_BOT[0], t))
        g = int(L(BG_TOP[1], BG_BOT[1], t))
        b = int(L(BG_TOP[2], BG_BOT[2], t))
        d.line([(0, y), (W, y)], fill=(r, g, b, 255))
    return img.convert("RGB")


def make_bg_with_blobs(blob_positions=None):
    """Background with soft colorful ambient blobs (Apple style)."""
    img = make_bg().convert("RGBA")

    if blob_positions:
        blob_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(blob_layer)
        for (bx, by, br, color, alpha) in blob_positions:
            for r in range(br, 0, -3):
                a = int(alpha * (r / br) ** 1.5)
                a = min(a, 60)
                bd.ellipse([bx - r, by - r, bx + r, by + r], fill=(*color, a))
        blob_layer = blob_layer.filter(ImageFilter.GaussianBlur(radius=60))
        img = Image.alpha_composite(img, blob_layer)

    return img.convert("RGB")


# ── Glassmorphism card ─────────────────────────────────
def glass_card(img, box, blur_radius=20, opacity=180, border_opacity=100):
    """Draw a frosted glass card with blur effect."""
    x0, y0, x1, y1 = box
    rgba = img.convert("RGBA")

    # Create the glass panel
    # 1. Crop the region behind the card
    region = rgba.crop((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
    # 2. Blur it heavily
    blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # 3. Overlay with semi-transparent white
    overlay = Image.new("RGBA", blurred.size, (255, 253, 248, opacity))
    glass = Image.alpha_composite(blurred, overlay)
    # 4. Paste back
    rgba.paste(glass, (max(0, x0), max(0, y0)))

    result = rgba.convert("RGB")
    d = ImageDraw.Draw(result)
    # Border
    d.rounded_rectangle(box, radius=20, fill=None,
                       outline=(255, 255, 255, border_opacity), width=1)
    return result, d


def glass_pill(img, box, opacity=200):
    """Small glass pill shape."""
    x0, y0, x1, y1 = box
    rgba = img.convert("RGBA")
    region = rgba.crop((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
    blurred = region.filter(ImageFilter.GaussianBlur(radius=15))
    overlay = Image.new("RGBA", blurred.size, (255, 253, 248, opacity))
    glass = Image.alpha_composite(blurred, overlay)
    rgba.paste(glass, (max(0, x0), max(0, y0)))
    result = rgba.convert("RGB")
    d = ImageDraw.Draw(result)
    d.rounded_rectangle(box, radius=(y1 - y0) // 2, fill=None,
                       outline=(255, 255, 255), width=1)
    return result, d


def soft_shadow(img, box, r=20, op=30):
    """Soft drop shadow."""
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    x0, y0, x1, y1 = box
    sd.rounded_rectangle((x0 + 3, y0 + 5, x1 + 3, y1 + 5), radius=16, fill=(0, 0, 0, op))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=r))
    return Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")


# ════════════════════════════════════════════════════════
#  SCENES
# ════════════════════════════════════════════════════════

def scene_01():
    """Title — faster, with ambient blobs."""
    print("  1. Title")

    blobs = [
        (W // 2 - 300, H // 2 - 100, 350, ACCENT, 30),
        (W // 2 + 250, H // 2 + 50, 280, PURPLE, 25),
        (W // 2, H // 2 + 200, 200, TEAL, 20),
    ]

    for i in range(35):  # Faster intro
        t = ease(i / 18)
        img = make_bg_with_blobs([(bx, by, int(br * t), c, int(a * t)) for bx, by, br, c, a in blobs])
        d = ImageDraw.Draw(img)

        y_off = int(L(-12, 0, t))
        dc(d, H // 2 - 55 + y_off, "WorldKit", sf_bold(92), LC(BG_TOP, DARK, t))

        st = ease(max(0, (i - 8) / 15))
        dc(d, H // 2 + 55, "The open-source world model SDK", sf_light(28), LC(BG_TOP, SECONDARY, st))

        save(img)

    hold(img, 22)

    # Accent line
    for i in range(12):
        t = ease(i / 12)
        img2 = img.copy()
        d = ImageDraw.Draw(img2)
        lw = int(60 * t)
        lx = W // 2 - lw // 2
        d.line([(lx, H // 2 + 38), (lx + lw, H // 2 + 38)], fill=LC(BG_TOP, ACCENT, t), width=2)
        save(img2)

    hold(img2, 15)
    fout(img2, 10)


def scene_02():
    """Hook — emotional, fast."""
    print("  2. Hook")

    blobs = [(W // 2, H // 2, 300, ACCENT, 15)]

    def build_a(t):
        img = make_bg_with_blobs([(bx, by, br, c, int(a * t)) for bx, by, br, c, a in blobs])
        d = ImageDraw.Draw(img)
        dc(d, H // 2 - 55, "World models predict the future.", sf_bold(52), LC(BG_TOP, DARK, t))
        st = ease(max(0, (t - 0.35) / 0.65))
        dc(d, H // 2 + 15, "Not the next word — the next state.", sf_light(28), LC(BG_TOP, SECONDARY, st))
        return img

    for i in range(30):
        save(build_a(ease(i / 18)))
    hold(build_a(1.0), 35)
    fout(build_a(1.0), 10)

    def build_b(t):
        img = make_bg_with_blobs([(W // 2, H // 2, 250, RED, int(18 * t))])
        d = ImageDraw.Draw(img)
        dc(d, H // 2 - 55, "But training them?", sf_bold(52), LC(BG_TOP, DARK, t))
        st = ease(max(0, (t - 0.3) / 0.7))
        dc(d, H // 2 + 15, "GPU clusters. Months of tuning. 6+ hyperparameters.", sf_light(26), LC(BG_TOP, RED, st))
        return img

    for i in range(25):
        save(build_b(ease(i / 16)))
    hold(build_b(1.0), 35)
    fout(build_b(1.0), 10)


def scene_03():
    """Until now — hero moment."""
    print("  3. Until now")

    blobs = [
        (W // 2 - 150, H // 2, 350, ACCENT, 35),
        (W // 2 + 200, H // 2 - 50, 250, PURPLE, 28),
    ]

    for i in range(30):
        t = ease(i / 18)
        img = make_bg_with_blobs([(bx, by, int(br * t), c, int(a * t)) for bx, by, br, c, a in blobs])
        d = ImageDraw.Draw(img)
        y_off = int(L(20, 0, t))
        dc(d, H // 2 - 40 + y_off, "Until now.", sf_bold(76), LC(BG_TOP, ACCENT, t))
        save(img)

    hold(img, 30)
    fout(img, 10)


def scene_04():
    """pip install — glass terminal card."""
    print("  4. Install")

    blobs = [
        (W // 2, H // 2 - 100, 300, ACCENT, 20),
        (W // 2 + 200, H // 2 + 100, 200, TEAL, 15),
    ]

    def build(t):
        img = make_bg_with_blobs(blobs)
        d = ImageDraw.Draw(img)

        card_w, card_h = 580, 90
        cx = W // 2 - card_w // 2
        cy = int(L(H // 2 - 25, H // 2 - 45, ease(t)))

        # Glass card
        img, d = glass_card(img, (cx, cy, cx + card_w, cy + card_h), blur_radius=25, opacity=200)

        # Dots
        for j, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            dot_c = LC(BG_TOP, c, t)
            dx = cx + 18 + j * 18
            dy = cy + 14
            d.ellipse([dx, dy, dx + 10, dy + 10], fill=dot_c)

        # Command
        ct = ease(max(0, (t - 0.15) / 0.6))
        d.text((cx + 30, cy + 35), "$ ", font=mono(30), fill=LC(BG_TOP, GREEN, ct))
        d.text((cx + 60, cy + 35), "pip install worldkit", font=mono(30), fill=LC(BG_TOP, DARK, ct))

        return img

    for i in range(28):
        save(build(min(1.0, i / 18)))
    hold(build(1.0), 45)
    fout(build(1.0), 10)


def scene_05():
    """Real data — grid with glass cards."""
    print("  5. Data")

    blobs = [(400, 500, 300, PURPLE, 12), (1500, 400, 250, TEAL, 10)]

    # Title
    def title(t):
        img = make_bg_with_blobs(blobs)
        d = ImageDraw.Draw(img)
        dc(d, H // 2 - 50, "Trained on real data.", sf_bold(56), LC(BG_TOP, DARK, t))
        st = ease(max(0, (t - 0.3) / 0.7))
        dc(d, H // 2 + 25, "200 expert demonstrations of robotic manipulation.", sf_light(24), LC(BG_TOP, SECONDARY, st))
        return img

    for i in range(25):
        save(title(ease(i / 16)))
    hold(title(1.0), 20)
    fout(title(1.0), 8)

    # Grid
    fs = 170
    gap = 16
    cols, rows = 6, 3
    tw_ = cols * fs + (cols - 1) * gap
    gx = (W - tw_) // 2
    gy = 200

    positions = []
    for r in range(rows):
        for c in range(cols):
            x = gx + c * (fs + gap)
            y = gy + r * (fs + gap)
            ep = r * 40 + c * 10
            ti = c * 2 + r
            positions.append((x, y, ep, ti))

    base_title = make_bg_with_blobs(blobs)
    bd = ImageDraw.Draw(base_title)
    dc(bd, 60, "Real Push-T Expert Demonstrations", sf(42), DARK)
    dc(bd, 115, "96×96 observations from a real robot", sf_light(22), SECONDARY)

    # Animate frames appearing
    for step in range(len(positions) + 5):
        img = base_title.copy()
        for idx in range(min(step + 1, len(positions))):
            x, y, ep, ti = positions[idx]
            img = soft_shadow(img, (x, y, x + fs, y + fs), r=12, op=18)
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((x - 1, y - 1, x + fs + 1, y + fs + 1),
                              radius=12, fill=WHITE, outline=TERTIARY, width=1)
            p = pt(ep, ti, sz=(fs, fs))
            img.paste(p, (x, y))
        save(img)

    hold(img, 35)

    # Filmstrip
    strip = make_bg_with_blobs(blobs)
    sd = ImageDraw.Draw(strip)
    dc(sd, 65, "A single episode", sf_bold(48), DARK)
    dc(sd, 125, "The model learns dynamics from observation alone.", sf_light(24), SECONDARY)

    ss = 165
    sg = 12
    nf = 8
    ts = nf * ss + (nf - 1) * sg
    sx = (W - ts) // 2
    sy = 240

    for i in range(nf):
        x = sx + i * (ss + sg)
        p = pt(0, i * 2, sz=(ss, ss))
        strip = soft_shadow(strip, (x, sy, x + ss, sy + ss), r=10, op=15)
        sd = ImageDraw.Draw(strip)
        sd.rounded_rectangle((x - 1, sy - 1, x + ss + 1, sy + ss + 1),
                           radius=10, fill=WHITE, outline=TERTIARY, width=1)
        strip.paste(p, (x, sy))
        dca(sd, x + ss // 2, sy + ss + 8, f"t={i * 2}", sf_light(16), SECONDARY)

    # Timeline with glass pill
    tly = sy + ss + 45
    sd.line([(sx + 30, tly), (sx + ts - 30, tly)], fill=ACCENT, width=2)
    for i in range(nf):
        dot_x = sx + i * (ss + sg) + ss // 2
        sd.ellipse([dot_x - 4, tly - 4, dot_x + 4, tly + 4], fill=ACCENT)

    # Glass bottom bar
    strip, sd = glass_card(strip, (W // 2 - 380, tly + 30, W // 2 + 380, tly + 90), blur_radius=20, opacity=210)
    dc(sd, tly + 45, "No physics engine. No simulator. Just observations.", sf_med(22), TEXT)

    from PIL import Image as PILImage
    crossfade_imgs(img, strip, 12)
    hold(strip, 45)
    fout(strip, 10)


def crossfade_imgs(a, b, n):
    for i in range(n):
        save(Image.blend(a, b, eio(i / n)))


def scene_06():
    """Encode — glass panels, clean visualization."""
    print("  6. Encode")

    np.random.seed(42)
    latent = np.random.randn(192).astype(np.float32) * 0.8

    blobs = [(350, 450, 250, PURPLE, 15), (1400, 350, 280, TEAL, 12)]

    def build(t):
        img = make_bg_with_blobs(blobs)
        d = ImageDraw.Draw(img)

        dc(d, 50, "Encode", sf_bold(52), LC(BG_TOP, DARK, t))
        dc(d, 115, "Any observation becomes a 192-dimensional latent vector.", sf_light(24), LC(BG_TOP, SECONDARY, ease(max(0, (t - 0.15) / 0.5))))

        # Image in glass card
        it = ease(min(1.0, t / 0.35))
        if it > 0:
            ix, iy = 180, 270
            p = pt(5, 3, sz=(320, 320))
            img = soft_shadow(img, (ix, iy, ix + 320, iy + 320), r=18, op=int(20 * it))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((ix - 2, iy - 2, ix + 322, iy + 322),
                              radius=14, fill=WHITE, outline=LC(BG_TOP, TERTIARY, it), width=1)
            img.paste(p, (ix, iy))
            dca(d, ix + 160, iy + 330, "96×96 observation", sf_light(18), LC(BG_TOP, SECONDARY, it))

        # Arrow
        at = ease(max(0, (t - 0.3) / 0.2))
        if at > 0:
            ax, ay = 560, 430
            aw = int(90 * at)
            d.line([(ax, ay), (ax + aw, ay)], fill=LC(BG_TOP, ACCENT, at), width=2)
            if aw > 25:
                d.polygon([(ax + aw, ay - 5), (ax + aw + 10, ay), (ax + aw, ay + 5)], fill=LC(BG_TOP, ACCENT, at))

            # Glass pill for "ViT Encoder"
            if at > 0.5:
                pill_t = ease((at - 0.5) / 0.5)
                img, d = glass_pill(img, (ax + 10, ay - 30, ax + aw - 10, ay - 8), opacity=int(210 * pill_t))
                dca(d, ax + aw // 2, ay - 28, "ViT", sf_light(16), LC(BG_TOP, TEAL, pill_t))

        # Latent bars in glass card
        bt = ease(max(0, (t - 0.45) / 0.55))
        if bt > 0:
            bx, by, bw, bh = 710, 240, 980, 300

            img, d = glass_card(img, (bx, by, bx + bw, by + bh + 30), blur_radius=22, opacity=int(210 * bt))
            d.text((bx + 15, by + 10), "z ∈ ℝ¹⁹²", font=mono(18), fill=LC(BG_TOP, ACCENT, bt))

            n_bars = int(192 * ease(bt))
            mid_y = by + 50 + bh // 2
            max_v = max(abs(v) for v in latent) or 1.0
            bw_ = max(2, bw // 192 - 1)

            for i in range(n_bars):
                v = latent[i]
                h = int((v / max_v) * (bh // 2 - 35) * bt)
                xi = bx + 10 + i * (bw_ + 1)
                c = ACCENT if v >= 0 else TEAL
                if h > 0:
                    d.rectangle([xi, mid_y - h, xi + bw_, mid_y], fill=LC(BG_TOP, c, bt))
                else:
                    d.rectangle([xi, mid_y, xi + bw_, mid_y - h], fill=LC(BG_TOP, c, bt))

            d.line([(bx + 10, mid_y), (bx + bw - 10, mid_y)], fill=LC(BG_TOP, TERTIARY, bt), width=1)

            # Stats
            d.text((bx + 15, by + bh - 5), "‖z‖ = 11.55", font=mono(16), fill=LC(BG_TOP, SECONDARY, bt))
            d.text((bx + 200, by + bh - 5), "range: [−2.10, +2.03]", font=mono(16), fill=LC(BG_TOP, SECONDARY, bt))

        # Code in glass pill
        ct = ease(max(0, (t - 0.75) / 0.25))
        if ct > 0:
            cy = 680
            img, d = glass_card(img, (220, cy, W - 220, cy + 65), blur_radius=18, opacity=int(220 * ct))
            d.text((260, cy + 18), "z = model.encode(frame)", font=mono(24), fill=LC(BG_TOP, DARK, ct))
            d.text((760, cy + 18), "# → (192,)", font=mono(24), fill=LC(BG_TOP, SECONDARY, ct))

        return img

    for i in range(45):
        save(build(min(1.0, i / 32)))
    hold(build(1.0), 40)
    fout(build(1.0), 10)


def scene_07():
    """Predict — trajectory with glass chart."""
    print("  7. Predict")

    norms = [10.96, 10.80, 10.56, 10.37, 10.19, 10.05, 9.88, 9.72, 9.61, 9.50]
    blobs = [(300, 400, 200, TEAL, 12), (1600, 300, 250, ACCENT, 10)]

    def build(t):
        img = make_bg_with_blobs(blobs)
        d = ImageDraw.Draw(img)

        dc(d, 50, "Predict", sf_bold(52), LC(BG_TOP, DARK, t))
        dc(d, 115, "Imagine the future — entirely in latent space.", sf_light(24), LC(BG_TOP, SECONDARY, ease(max(0, (t - 0.1) / 0.4))))

        # Frame sequence
        for i in range(5):
            ft = ease(max(0, min(1.0, (t - i * 0.05) / 0.2)))
            if ft <= 0:
                continue
            p = pt(0, i * 3, sz=(145, 145))
            x = 90 + i * 162
            y = 240
            img = soft_shadow(img, (x, y, x + 145, y + 145), r=10, op=int(15 * ft))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((x - 1, y - 1, x + 146, y + 146),
                              radius=10, fill=WHITE, outline=LC(BG_TOP, TERTIARY, ft), width=1)
            img.paste(p, (x, y))
            dca(d, x + 72, y + 150, f"t+{i}", sf_light(14), LC(BG_TOP, SECONDARY, ft))

        # Chart in glass card
        cht = ease(max(0, (t - 0.35) / 0.65))
        if cht > 0:
            cx, cy, cw, ch = 960, 225, 760, 220
            img, d = glass_card(img, (cx, cy, cx + cw, cy + ch + 35), blur_radius=20, opacity=int(215 * cht))
            d.text((cx + 15, cy + 8), "Latent trajectory", font=sf_light(16), fill=LC(BG_TOP, SECONDARY, cht))

            min_n, max_n = 9.2, 11.2
            for g in range(5):
                gy = cy + 30 + int(g * ch / 4)
                d.line([(cx + 10, gy), (cx + cw - 10, gy)], fill=LC(BG_TOP, (235, 235, 240), cht), width=1)

            n_pts = int(len(norms) * ease(cht))
            pts = []
            for i in range(len(norms)):
                px = cx + 30 + int(i * (cw - 60) / (len(norms) - 1))
                py = cy + 30 + int((1.0 - (norms[i] - min_n) / (max_n - min_n)) * ch)
                pts.append((px, py))

            for i in range(min(n_pts - 1, len(pts) - 1)):
                d.line([pts[i], pts[i + 1]], fill=LC(BG_TOP, ACCENT, cht), width=3)
            for i in range(n_pts):
                px, py = pts[i]
                d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=LC(BG_TOP, ACCENT, cht))

            if cht > 0.8:
                d.text((cx + cw - 120, cy + 35), "drift: 1.62", font=mono(16), fill=LC(BG_TOP, GREEN, (cht - 0.8) / 0.2))

        # Code
        ct = ease(max(0, (t - 0.7) / 0.3))
        if ct > 0:
            cby = 540
            img, d = glass_card(img, (90, cby, W - 90, cby + 100), blur_radius=18, opacity=int(220 * ct))
            d.text((140, cby + 15), "result = model.predict(frame, actions)", font=mono(22), fill=LC(BG_TOP, DARK, ct))
            d.text((140, cby + 52), "# result.latent_trajectory → (10, 192)", font=mono(22), fill=LC(BG_TOP, SECONDARY, ct))

        return img

    for i in range(48):
        save(build(min(1.0, i / 34)))
    hold(build(1.0), 40)
    fout(build(1.0), 10)


def scene_08():
    """Plan — current → goal with glass CEM pill."""
    print("  8. Plan")

    blobs = [(350, 450, 220, ACCENT, 12), (1550, 400, 200, GREEN, 10)]

    def build(t):
        img = make_bg_with_blobs(blobs)
        d = ImageDraw.Draw(img)

        dc(d, 50, "Plan", sf_bold(52), LC(BG_TOP, DARK, t))
        dc(d, 115, "Find optimal actions to reach any goal.", sf_light(24), LC(BG_TOP, SECONDARY, ease(max(0, (t - 0.1) / 0.4))))

        y1 = 280
        # Current
        ct_ = ease(min(1.0, t / 0.3))
        p1 = pt(0, 0, sz=(300, 300))
        x1 = 220
        img = soft_shadow(img, (x1, y1, x1 + 300, y1 + 300), r=16, op=int(18 * ct_))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((x1 - 2, y1 - 2, x1 + 302, y1 + 302), radius=14, fill=WHITE,
                          outline=LC(BG_TOP, ACCENT, ct_), width=2)
        img.paste(p1, (x1, y1))
        dca(d, x1 + 150, y1 + 315, "Current", sf_med(20), LC(BG_TOP, SECONDARY, ct_))

        # Goal
        gt = ease(max(0, (t - 0.15) / 0.3))
        p2 = pt(80, 14, sz=(300, 300))
        x2 = W - 520
        img = soft_shadow(img, (x2, y1, x2 + 300, y1 + 300), r=16, op=int(18 * gt))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((x2 - 2, y1 - 2, x2 + 302, y1 + 302), radius=14, fill=WHITE,
                          outline=LC(BG_TOP, GREEN, gt), width=2)
        img.paste(p2, (x2, y1))
        dca(d, x2 + 150, y1 + 315, "Goal", sf_med(20), LC(BG_TOP, GREEN, gt))

        # CEM glass pill in center
        mt = ease(max(0, (t - 0.4) / 0.3))
        if mt > 0:
            mid = W // 2
            my = y1 + 120
            lc_ = LC(BG_TOP, ACCENT, mt)
            d.line([(x1 + 310, my + 20), (mid - 80, my + 20)], fill=lc_, width=2)
            d.line([(mid + 80, my + 20), (x2 - 10, my + 20)], fill=lc_, width=2)
            d.polygon([(x2 - 10, my + 14), (x2, my + 20), (x2 - 10, my + 26)], fill=lc_)

            img, d = glass_pill(img, (mid - 75, my, mid + 75, my + 40), opacity=int(220 * mt))
            dca(d, mid, my + 8, "CEM Planner", sf_med(18), LC(BG_TOP, ACCENT, mt))

        # Code
        code_t = ease(max(0, (t - 0.7) / 0.3))
        if code_t > 0:
            cby = 700
            img, d = glass_card(img, (250, cby, W - 250, cby + 60), blur_radius=18, opacity=int(220 * code_t))
            d.text((290, cby + 16), "plan = model.plan(current, goal, max_steps=50)", font=mono(22), fill=LC(BG_TOP, DARK, code_t))

        return img

    for i in range(48):
        save(build(min(1.0, i / 34)))
    hold(build(1.0), 40)
    fout(build(1.0), 10)


def scene_09():
    """Stats — big numbers, staggered."""
    print("  9. Stats")

    stats = [
        ("13M", "parameters", ACCENT),
        ("1", "hyperparameter", TEAL),
        ("60s", "to train", GREEN),
        ("50MB", "model size", ORANGE),
    ]

    blobs = [
        (W // 4, H // 2, 200, ACCENT, 15),
        (W * 3 // 4, H // 2, 200, GREEN, 12),
    ]

    def build(t):
        img = make_bg_with_blobs(blobs)
        d = ImageDraw.Draw(img)

        col_w = W // len(stats)
        for i, (num, label, color) in enumerate(stats):
            st = ease(max(0, min(1.0, (t - i * 0.1) / 0.3)))
            cx_ = col_w * i + col_w // 2
            y_off = int(L(25, 0, st))
            dca(d, cx_, H // 2 - 75 + y_off, num, sf_bold(80), LC(BG_TOP, color, st))
            lt = ease(max(0, (st - 0.5) / 0.5))
            dca(d, cx_, H // 2 + 25, label, sf_light(22), LC(BG_TOP, SECONDARY, lt))

        # Glass bottom bar
        bt = ease(max(0, (t - 0.65) / 0.35))
        if bt > 0:
            img, d = glass_card(img, (W // 2 - 350, H - 190, W // 2 + 350, H - 140), blur_radius=20, opacity=int(215 * bt))
            dc(d, H - 180, "One loss. One hyperparameter. No collapse.", sf_med(22), LC(BG_TOP, ACCENT, bt))

        return img

    for i in range(40):
        save(build(min(1.0, i / 28)))
    hold(build(1.0), 50)
    fout(build(1.0), 12)


def scene_10():
    """Closing — WorldKit, install, links."""
    print("  10. Closing")

    blobs = [
        (W // 2 - 200, 300, 350, ACCENT, 25),
        (W // 2 + 250, 350, 280, PURPLE, 20),
        (W // 2, 500, 200, TEAL, 15),
    ]

    def build(t):
        img = make_bg_with_blobs([(bx, by, int(br * t), c, int(a * t)) for bx, by, br, c, a in blobs])
        d = ImageDraw.Draw(img)

        tt = ease(min(1.0, t / 0.25))
        dc(d, 200, "WorldKit", sf_bold(84), LC(BG_TOP, DARK, tt))

        # Accent line
        lt = ease(max(0, (t - 0.15) / 0.15))
        lw = int(50 * lt)
        lx = W // 2 - lw // 2
        d.line([(lx, 290), (lx + lw, 290)], fill=LC(BG_TOP, ACCENT, lt), width=2)

        # Install button — glass pill
        bt = ease(max(0, (t - 0.25) / 0.2))
        if bt > 0:
            bx0, by0 = W // 2 - 210, 330
            bx1, by1 = W // 2 + 210, 390
            d.rounded_rectangle((bx0, by0, bx1, by1), radius=30, fill=LC(BG_TOP, ACCENT, bt))
            dc(d, 345, "pip install worldkit", mono(26), LC(ACCENT, WHITE, bt))

        # Links
        links = [
            ("github.com/DilpreetBansi/worldkit", ACCENT),
            ("huggingface.co/DilpreetBansi/pusht-base", TEAL),
            ("pypi.org/project/worldkit", SECONDARY),
        ]
        for j, (url, c) in enumerate(links):
            ut = ease(max(0, min(1.0, (t - 0.45 - j * 0.06) / 0.2)))
            dc(d, 440 + j * 38, url, sf_reg(20), LC(BG_TOP, c, ut))

        at = ease(max(0, (t - 0.8) / 0.2))
        dc(d, H - 110, "Built by Dilpreet Bansi", sf_med(22), LC(BG_TOP, SECONDARY, at))
        dc(d, H - 78, "MIT License", sf_light(16), LC(BG_TOP, TERTIARY, at))

        return img

    for i in range(42):
        save(build(min(1.0, i / 28)))
    hold(build(1.0), 65)
    fout(build(1.0), 20)


def main():
    global FI
    FI = 0
    print("Generating Apple-style product film...")
    scene_01()
    scene_02()
    scene_03()
    scene_04()
    scene_05()
    scene_06()
    scene_07()
    scene_08()
    scene_09()
    scene_10()

    print(f"\nTotal: {FI} frames ({FI / FPS:.0f}s at {FPS}fps)")

    out_mp4 = Path(__file__).parent.parent / "demo.mp4"
    out_gif = Path(__file__).parent.parent / "demo.gif"

    print("Encoding MP4 (1080p, high quality)...")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(OUT_DIR / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "14", "-preset", "slow", "-movflags", "+faststart",
        "-tune", "animation",
        out_mp4,
    ], capture_output=True)

    print("Encoding GIF (720p)...")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(OUT_DIR / "frame_%05d.png"),
        "-vf", "fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=164:stats_mode=diff[p];[s1][p]paletteuse=dither=floyd_steinberg",
        out_gif,
    ], capture_output=True)

    print(f"\n  MP4: {out_mp4} ({out_mp4.stat().st_size / 1e6:.1f} MB)")
    print(f"  GIF: {out_gif} ({out_gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
