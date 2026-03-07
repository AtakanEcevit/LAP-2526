"""
Hero Banner GIF Generator for README.
Generates an animated banner showing biometric verification across 3 modalities.

Usage: python docs/generate_banner.py
Output: docs/assets/hero_banner.gif
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Configuration ──────────────────────────────────────────────
WIDTH, HEIGHT = 900, 320
BG_TOP = (13, 17, 23)        # #0d1117  (GitHub dark)
BG_BOT = (22, 27, 34)        # #161b22
ACCENT = (88, 166, 255)       # Blue accent
CYAN = (56, 239, 125)         # Green for "Verified"
SCAN_COLOR = (0, 200, 255)    # Cyan scan line
WHITE = (255, 255, 255)
DIM = (136, 136, 136)         # Dimmed text
BAR_BG = (48, 54, 61)
BAR_FILL = (88, 166, 255)
FPS = 15
FRAMES_PER_PHASE = 30         # 30 frames per modality
PAUSE_FRAMES = 8              # Pause on "VERIFIED" result

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")
OUTPUT_PATH = os.path.join(ASSETS_DIR, "hero_banner.gif")

MODALITIES = [
    {
        "label": "SIGNATURE VERIFICATION",
        "icon": "\u270d\ufe0f",
        "image": os.path.join(ASSETS_DIR, "sample_signature.png"),
        "score": 97.3,
    },
    {
        "label": "FACE VERIFICATION",
        "icon": "\ud83d\udc64",
        "image": os.path.join(ASSETS_DIR, "sample_face.png"),
        "score": 99.1,
    },
    {
        "label": "FINGERPRINT VERIFICATION",
        "icon": "\ud83d\udd90\ufe0f",
        "image": os.path.join(ASSETS_DIR, "sample_fingerprint.png"),
        "score": 98.6,
    },
]


def make_gradient_bg():
    """Create a vertical gradient background image."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))
    return img


def load_sample_image(path, size=(160, 160)):
    """Load and resize a sample biometric image with rounded corners."""
    img = Image.open(path).convert("RGBA")
    img = img.resize(size, Image.LANCZOS)

    # Create rounded corners mask
    mask = Image.new("L", size, 0)
    mask_draw = ImageDraw.Draw(mask)
    radius = 12
    mask_draw.rounded_rectangle([(0, 0), (size[0] - 1, size[1] - 1)],
                                 radius=radius, fill=255)
    img.putalpha(mask)
    return img


def get_font(size):
    """Get a font, falling back to default."""
    # Try common Windows fonts
    font_paths = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def get_bold_font(size):
    """Get a bold font, fall back to regular."""
    font_paths = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return get_font(size)


def draw_scan_line(img, progress, image_box):
    """Draw a glowing cyan scan line over the sample image area."""
    x1, y1, x2, y2 = image_box
    scan_y = int(y1 + (y2 - y1) * progress)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Main scan line
    draw.line([(x1, scan_y), (x2, scan_y)], fill=(*SCAN_COLOR, 200), width=2)

    # Glow effect (multiple fading lines)
    for offset in range(1, 8):
        alpha = max(0, 200 - offset * 30)
        if scan_y - offset >= y1:
            draw.line([(x1, scan_y - offset), (x2, scan_y - offset)],
                      fill=(*SCAN_COLOR, alpha), width=1)
        if scan_y + offset <= y2:
            draw.line([(x1, scan_y + offset), (x2, scan_y + offset)],
                      fill=(*SCAN_COLOR, alpha), width=1)

    return Image.alpha_composite(img.convert("RGBA"), overlay)


def draw_progress_bar(draw, x, y, width, height, progress, score):
    """Draw a rounded progress bar with percentage text."""
    # Background
    draw.rounded_rectangle([(x, y), (x + width, y + height)],
                            radius=height // 2, fill=BAR_BG)
    # Fill
    fill_w = max(height, int(width * progress))
    draw.rounded_rectangle([(x, y), (x + fill_w, y + height)],
                            radius=height // 2, fill=BAR_FILL)
    # Percentage text
    pct = progress * score
    font = get_font(14)
    text = f"{pct:.1f}%"
    draw.text((x + width + 12, y + 1), text, fill=WHITE, font=font)


def draw_border_glow(img, image_box, color, alpha=80):
    """Draw a subtle glow border around the sample image."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = image_box
    for i in range(4):
        a = max(0, alpha - i * 20)
        draw.rounded_rectangle(
            [(x1 - i - 1, y1 - i - 1), (x2 + i + 1, y2 + i + 1)],
            radius=12 + i, outline=(*color, a), width=1
        )
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def generate_phase_frames(bg, modality, phase_idx):
    """Generate all frames for one modality phase."""
    frames = []
    sample_img = load_sample_image(modality["image"])

    # Layout constants
    img_x, img_y = 80, 95
    img_w, img_h = 160, 160
    image_box = (img_x, img_y, img_x + img_w, img_y + img_h)

    text_x = 300
    bar_x = 300
    bar_y = 200
    bar_w = 400
    bar_h = 16

    title_font = get_bold_font(28)
    label_font = get_bold_font(18)
    sub_font = get_font(14)
    result_font = get_bold_font(22)

    scan_frames = 18
    fill_frames = 8
    result_frames = PAUSE_FRAMES

    # Adjust frame distribution
    total = scan_frames + fill_frames + result_frames
    if total > FRAMES_PER_PHASE:
        result_frames = FRAMES_PER_PHASE - scan_frames - fill_frames

    for i in range(FRAMES_PER_PHASE):
        frame = bg.copy().convert("RGBA")
        draw = ImageDraw.Draw(frame)

        # ── Title (always visible) ──
        draw.text((WIDTH // 2, 28), "Biometric Few-Shot Verification",
                  fill=WHITE, font=title_font, anchor="mt")
        draw.text((WIDTH // 2, 58), "Deep Metric Learning Framework",
                  fill=DIM, font=sub_font, anchor="mt")

        # ── Modality dots (bottom) ──
        dot_y = HEIGHT - 25
        for j in range(3):
            dot_x = WIDTH // 2 - 20 + j * 20
            color = ACCENT if j == phase_idx else (60, 60, 60)
            draw.ellipse([(dot_x - 4, dot_y - 4), (dot_x + 4, dot_y + 4)],
                         fill=color)

        # ── Paste sample image ──
        frame.paste(sample_img, (img_x, img_y), sample_img)

        # ── Modality label ──
        draw.text((text_x, 100), modality["label"],
                  fill=ACCENT, font=label_font)

        # ── Phase animation ──
        if i < scan_frames:
            # Scanning phase
            progress = i / scan_frames
            frame = draw_scan_line(frame, progress, image_box)
            frame = draw_border_glow(frame, image_box, SCAN_COLOR,
                                      alpha=int(60 + 40 * math.sin(progress * math.pi)))
            draw = ImageDraw.Draw(frame)

            draw.text((text_x, 135), "Analyzing biometric features...",
                      fill=DIM, font=sub_font)
            draw_progress_bar(draw, bar_x, bar_y, bar_w, bar_h,
                              progress * 0.85, modality["score"])

        elif i < scan_frames + fill_frames:
            # Progress fill phase
            fill_i = i - scan_frames
            progress = 0.85 + 0.15 * (fill_i / fill_frames)
            frame = draw_border_glow(frame, image_box, ACCENT, alpha=60)
            draw = ImageDraw.Draw(frame)

            draw.text((text_x, 135), "Computing similarity score...",
                      fill=DIM, font=sub_font)
            draw_progress_bar(draw, bar_x, bar_y, bar_w, bar_h,
                              progress, modality["score"])

        else:
            # Result phase
            frame = draw_border_glow(frame, image_box, CYAN, alpha=100)
            draw = ImageDraw.Draw(frame)

            draw_progress_bar(draw, bar_x, bar_y, bar_w, bar_h,
                              1.0, modality["score"])

            # Verified result
            result_text = f"VERIFIED  \u00b7  {modality['score']:.1f}% Match"
            draw.text((text_x, 245), result_text, fill=CYAN, font=result_font)

            draw.text((text_x, 135), "Identity confirmed",
                      fill=CYAN, font=sub_font)

        # Convert to RGB for GIF
        rgb_frame = Image.new("RGB", frame.size, BG_TOP)
        rgb_frame.paste(frame, mask=frame.split()[3])
        frames.append(rgb_frame)

    return frames


def main():
    print("Generating hero banner...")
    os.makedirs(ASSETS_DIR, exist_ok=True)

    bg = make_gradient_bg()

    all_frames = []
    for idx, mod in enumerate(MODALITIES):
        print(f"  Phase {idx + 1}/3: {mod['label']}...")
        phase_frames = generate_phase_frames(bg, mod, idx)
        all_frames.extend(phase_frames)

    # Save as GIF
    print(f"  Saving {len(all_frames)} frames to {OUTPUT_PATH}...")
    durations = []
    for i in range(len(all_frames)):
        frame_in_phase = i % FRAMES_PER_PHASE
        if frame_in_phase >= FRAMES_PER_PHASE - PAUSE_FRAMES:
            durations.append(120)  # Slower on result
        else:
            durations.append(1000 // FPS)

    all_frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=all_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"  Done! File size: {file_size / 1024:.0f} KB")
    if file_size > 2 * 1024 * 1024:
        print("  WARNING: File exceeds 2 MB, may load slowly on GitHub.")


if __name__ == "__main__":
    main()
