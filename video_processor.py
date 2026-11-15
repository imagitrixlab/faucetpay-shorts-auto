import os
import re
import sys
import glob
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# আপনার Google Drive ওপেন লিংকটি এখানে সেট করা আছে:
GDRIVE_SCREEN_LINK = "https://drive.google.com/file/d/1-2TCd-DvSQ2UZHViuINtPJkhc6SCAI5W/view?usp=sharing"

# Pillow 10+ compatibility fix for MoviePy 1.0.3
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, VideoClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips
)

W, H = 1080, 1920
VALID_AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac")
VALID_MEDIA_EXTS = VALID_AUDIO_EXTS + (".mp4", ".webm", ".mov")
VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VALID_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv")

def scan_all_starter_assets():
    """Scans root & assets/ folder and inventories every FaucetPay Starter Kit file."""
    inventory = {
        "logos_dark": [],
        "logos_light": [],
        "icons": [],
        "sonic": [],
        "sfx_coin": [],
        "sfx_other": [],
        "broll": [],
    }
    ignored_prefixes = ("final_faucetpay_shorts", "voice_", "bg_pulse", "room_air", "tts_", "seg_", "broll_bg", "screen_record")

    for folder in (".", "assets"):
        if not os.path.isdir(folder):
            continue
        for root_dir, _, files in os.walk(folder):
            for fname in sorted(files):
                fpath = os.path.join(root_dir, fname)
                low = fname.lower()
                if low.startswith(ignored_prefixes):
                    continue
                if not low.startswith("fp-"):
                    continue

                if low.endswith(VALID_IMAGE_EXTS):
                    if "icon" in low or "badge" in low or "mark" in low:
                        inventory["icons"].append(fpath)
                    elif "light" in low:
                        inventory["logos_light"].append(fpath)
                    else:
                        inventory["logos_dark"].append(fpath)
                elif low.endswith(VALID_AUDIO_EXTS):
                    if "sonic" in low:
                        inventory["sonic"].append(fpath)
                    elif "coin" in low or "drip" in low or "cascade" in low:
                        inventory["sfx_coin"].append(fpath)
                    else:
                        inventory["sfx_other"].append(fpath)
                elif low.endswith(VALID_VIDEO_EXTS):
                    inventory["broll"].append(fpath)

    print("📦 FaucetPay Starter Kit Inventory Detected:")
    for k, v in inventory.items():
        print(f"   • {k}: {len(v)} file(s) -> {v}")
    return inventory

def get_screen_recording():
    """Downloads the screen recording from Google Drive open link cleanly without --fuzzy."""
    link = GDRIVE_SCREEN_LINK.strip()
    if link.startswith("http"):
        print("☁️ Downloading screen recording from Google Drive open link...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)

        match = re.search(r"/d/([a-zA-Z0-9_-]+)", link) or re.search(r"id=([a-zA-Z0-9_-]+)", link)
        if match:
            file_id = match.group(1)
            subprocess.run(
                [sys.executable, "-m", "gdown", f"https://drive.google.com/uc?id={file_id}", "-O", "screen_record.mp4"],
                check=False
            )
            if not (os.path.exists("screen_record.mp4") and os.path.getsize("screen_record.mp4") > 5000000):
                subprocess.run(
                    [sys.executable, "-m", "gdown", file_id, "-O", "screen_record.mp4"],
                    check=True
                )
        else:
            subprocess.run(
                [sys.executable, "-m", "gdown", link, "-O", "screen_record.mp4"],
                check=True
            )

        if os.path.exists("screen_record.mp4") and os.path.getsize("screen_record.mp4") > 5000000:
            print("✅ Screen Recording Ready -> screen_record.mp4")
            return "screen_record.mp4"
        raise RuntimeError("❌ Google Drive download failed or file too small!")

    for pat in ("screen_record*", "Screen_Recording*", "60317*"):
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    raise FileNotFoundError("❌ Screen recording not found!")

def find_user_voice_file(screen_path):
    """Automatically selects the latest uploaded voice audio file in root or assets/."""
    screen_base = os.path.basename(screen_path).lower()
    excluded_prefixes = (
        "final_faucetpay_shorts", "voice_clean", "voice_human", "bg_pulse",
        "room_air", "tts_", "fp-", "60317", "screen_record", "screen_recording", "seg_", "broll_bg"
    )

    audio_candidates = []
    media_candidates = []
    for folder in (".", "assets"):
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath):
                continue
            low = fname.lower()
            if low == screen_base or low.startswith(excluded_prefixes):
                continue
            if low.endswith(VALID_AUDIO_EXTS):
                audio_candidates.append((os.path.getmtime(fpath), fpath))
            elif low.endswith(VALID_MEDIA_EXTS):
                media_candidates.append((os.path.getmtime(fpath), fpath))

    if audio_candidates:
        audio_candidates.sort(reverse=True)
        chosen = audio_candidates[0][1]
        print(f"🎙️ Selected User Voice Audio -> {chosen}")
        return chosen
    if media_candidates:
        media_candidates.sort(reverse=True)
        chosen = media_candidates[0][1]
        print(f"🎙️ Selected User Voice Media -> {chosen}")
        return chosen

    raise FileNotFoundError("❌ No user voice/audio file found! Upload your .wav audio file to GitHub.")

def get_video_info(video_path):
    """Gets dimensions, clean crop box, and duration via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    raw_dur = (frame_count / fps) if (fps > 0 and frame_count > 0) else 83.0
    if not ret:
        return (0, 80, 1080, 2280), raw_dur

    fh, fw, _ = frame.shape
    if fh > fw:
        y1 = int(fh * 0.042)
        y2 = int(fh * 0.962)
        print(f"📱 Native Portrait Capture ({fw}x{fh}, {raw_dur:.1f}s) -> Crop (0, {y1}, {fw}, {y2})")
        return (0, y1, fw, y2), raw_dur

    return (int(fw * 0.406), int(fh * 0.190), int(fw * 0.594), int(fh * 0.943)), raw_dur

# Tutorial Stage Geometry: Step Header (y=88..200), Phone (y=220..1460), Bottom Caption (y=1485..1695)
PHONE_W, PHONE_H = 660, 1240
PHONE_X = (W - PHONE_W) // 2
PHONE_Y = 220

def build_studio_base_and_mask():
    """Builds the base emerald-navy studio stage and phone bezel mask."""
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.zeros((H, W, 3), dtype=np.float32)
    bg[:, :, 0] = 5 + (y_grid / H) * 7
    bg[:, :, 1] = 20 + (y_grid / H) * 22
    bg[:, :, 2] = 26 + (y_grid / H) * 26

    dist_center = np.sqrt(((x_grid - W * 0.5) / 540.0) ** 2 + ((y_grid - H * 0.45) / 620.0) ** 2)
    glow = np.exp(-dist_center ** 2) * 38.0
    bg[:, :, 1] += glow * 1.10
    bg[:, :, 2] += glow * 0.85

    base_rgb = np.clip(bg, 0, 255).astype(np.uint8)

    # Bezel overlay (RGBA) drawn AFTER optional blurred B-Roll backdrop texture
    bezel = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bezel)
    draw.rounded_rectangle(
        [PHONE_X - 16, PHONE_Y - 14, PHONE_X + PHONE_W + 16, PHONE_Y + PHONE_H + 18],
        radius=46, fill=(2, 10, 12, 210)
    )
    draw.rounded_rectangle(
        [PHONE_X - 7, PHONE_Y - 7, PHONE_X + PHONE_W + 7, PHONE_Y + PHONE_H + 7],
        radius=38, outline=(40, 235, 165, 240), width=4
    )
    bezel_np = np.array(bezel)
    bezel_rgb = bezel_np[:, :, :3].astype(np.float32)
    bezel_alpha = (bezel_np[:, :, 3:4].astype(np.float32)) / 255.0

    mask_img = Image.new("L", (PHONE_W, PHONE_H), 0)
    ImageDraw.Draw(mask_img).rounded_rectangle([0, 0, PHONE_W - 1, PHONE_H - 1], radius=32, fill=255)
    screen_alpha = (np.array(mask_img).astype(np.float32) / 255.0)[:, :, None]

    return base_rgb, bezel_rgb, bezel_alpha, screen_alpha

BASE_STAGE_RGB, BEZEL_RGB, BEZEL_ALPHA, SCREEN_ALPHA = build_studio_base_and_mask()

def prepare_broll_backdrop_reader(broll_list, total_dur):
    """
    Safely utilizes Starter Kit B-Roll clips as a heavily blurred, dark ambient stage texture
    (14% opacity behind the phone bezel) so 100% of starter assets are used with ZERO stock-clip penalty.
    """
    if not broll_list:
        return None
    broll_bg_path = "broll_bg.mp4"
    src = broll_list[0]
    subprocess.run([
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", src,
        "-t", f"{total_dur:.2f}",
        "-vf", f"fps=30,scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:4,eq=brightness=-0.30:saturation=0.55",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-an",
        broll_bg_path
    ], check=True)
    print(f"🎨 Integrated Starter Kit B-Roll Ambient Backdrop -> {os.path.basename(src)}")
    return VideoFileClip(broll_bg_path).without_audio().set_fps(30)

def prepare_crystal_phone_clip(screen_path, seg_idx, start_t, end_t, raw_dur, crop_box, target_dur, stage_offset_t=0.0, broll_clip=None):
    """Pre-trims & converts each segment to 30fps CFR via native FFmpeg (~90s total render)."""
    safe_start = min(start_t, max(0.0, raw_dur - 1.0))
    safe_end = min(end_t, max(safe_start + 0.5, raw_dur - 0.05))
    src_dur = max(0.5, safe_end - safe_start)
    pts_factor = target_dur / src_dur

    x1, y1, x2, y2 = crop_box
    cw = max(2, (x2 - x1) // 2 * 2)
    ch = max(2, (y2 - y1) // 2 * 2)

    seg_file = f"seg_{seg_idx}.mp4"
    vf = (
        f"setpts={pts_factor:.6f}*PTS,"
        f"fps=30,"
        f"crop={cw}:{ch}:{x1}:{y1},"
        f"scale={PHONE_W}:{PHONE_H}:flags=lanczos,"
        f"unsharp=3:3:0.52:3:3:0.0,"
        f"eq=contrast=1.03:brightness=0.01"
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{safe_start:.3f}",
        "-t", f"{src_dur:.3f}",
        "-i", screen_path,
        "-vf", vf,
        "-t", f"{target_dur:.3f}",
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "15",
        "-pix_fmt", "yuv420p",
        "-an",
        seg_file
    ], check=True)

    sub = VideoFileClip(seg_file).without_audio().set_fps(30)

    def composite_on_stage(get_frame, t):
        screen = get_frame(t)
        if screen.shape[0] != PHONE_H or screen.shape[1] != PHONE_W:
            screen = cv2.resize(screen, (PHONE_W, PHONE_H), interpolation=cv2.INTER_LANCZOS4)

        canvas_f = BASE_STAGE_RGB.astype(np.float32)
        if broll_clip is not None:
            bt = min(stage_offset_t + t, max(0.0, broll_clip.duration - 0.05))
            bframe = broll_clip.get_frame(bt).astype(np.float32)
            canvas_f = canvas_f * 0.86 + bframe * 0.14

        # Apply titanium phone bezel
        canvas_f = BEZEL_RGB * BEZEL_ALPHA + canvas_f * (1.0 - BEZEL_ALPHA)
        canvas = np.clip(canvas_f, 0, 255).astype(np.uint8)

        # Composite crystal-clear mobile screen inside bezel
        roi = canvas[PHONE_Y:PHONE_Y + PHONE_H, PHONE_X:PHONE_X + PHONE_W].astype(np.float32)
        blended = screen.astype(np.float32) * SCREEN_ALPHA + roi * (1.0 - SCREEN_ALPHA)
        canvas[PHONE_Y:PHONE_Y + PHONE_H, PHONE_X:PHONE_X + PHONE_W] = blended.astype(np.uint8)
        return canvas

    return sub.fl(composite_on_stage).set_duration(target_dur)

def fit_font(draw, text, max_w, start_size):
    """Automatically scales font down so zero letters ever touch box borders."""
    for sz in range(start_size, 18, -2):
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
        except Exception:
            return ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=f)
        if (bbox[2] - bbox[0]) <= max_w:
            return f
    return ImageFont.load_default()

def build_official_logo_card(logo_path, icon_path, duration, is_outro=False):
    """Creates a 100% original animated card using FaucetPay's official PNG logo & icon."""
    pil_card = Image.fromarray(BASE_STAGE_RGB.copy()).convert("RGBA")
    draw = ImageDraw.Draw(pil_card)

    draw.rounded_rectangle([75, 500, W - 75, 1400], radius=44, fill=(6, 26, 28, 248), outline=(40, 235, 165, 255), width=5)

    if logo_path and os.path.exists(logo_path):
        logo_img = Image.open(logo_path).convert("RGBA")
        lw = 640
        lh = int(logo_img.height * (lw / logo_img.width))
        logo_img = logo_img.resize((lw, lh), Image.LANCZOS)
        pil_card.paste(logo_img, ((W - lw) // 2, 625), logo_img)

    if icon_path and os.path.exists(icon_path):
        ic = Image.open(icon_path).convert("RGBA").resize((84, 84), Image.LANCZOS)
        pil_card.paste(ic, ((W - 84) // 2, 1145), ic)

    if is_outro:
        line1 = "TRY FAUCETPAY COIN SWAP"
        line2 = "OFFICIAL LINK IN DESCRIPTION ↓"
    else:
        line1 = "FAUCETPAY COIN SWAP TUTORIAL"
        line2 = "HOW TO CONVERT CRYPTO IN 3 EASY STEPS"

    f_title = fit_font(draw, line1, W - 210, 42)
    f_sub = fit_font(draw, line2, W - 210, 32)

    b1 = draw.textbbox((0, 0), line1, font=f_title)
    b2 = draw.textbbox((0, 0), line2, font=f_sub)
    draw.text(((W - (b1[2] - b1[0])) // 2, 935), line1, font=f_title, fill=(255, 255, 255, 255))
    draw.text(((W - (b2[2] - b2[0])) // 2, 1020), line2, font=f_sub, fill=(255, 225, 55, 255))

    card_rgb = np.array(pil_card.convert("RGB"))
    return ImageClip(card_rgb).set_duration(duration).set_fps(30)

def make_step_header(step_tag, bot_text, start_t, duration, icon_path=None):
    """Renders a crystal-clear STEP 1/2/3 tutorial header cleanly above the phone frame."""
    bw, bh = 880, 114
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([4, 4, bw - 4, bh - 4], radius=28, fill=(5, 28, 30, 246), outline=(255, 215, 65, 255), width=4)

    text_offset_x = 0
    if icon_path and os.path.exists(icon_path):
        ic = Image.open(icon_path).convert("RGBA").resize((64, 64), Image.LANCZOS)
        img.paste(ic, (28, (bh - 64) // 2), ic)
        text_offset_x = 34

    avail_w = bw - 120 - text_offset_x
    f_top = fit_font(draw, step_tag, avail_w, 27)
    f_bot = fit_font(draw, bot_text, avail_w, 33)

    b1 = draw.textbbox((0, 0), step_tag, font=f_top)
    b2 = draw.textbbox((0, 0), bot_text, font=f_bot)
    draw.text(((bw - (b1[2] - b1[0])) // 2 + text_offset_x // 2, 17), step_tag, font=f_top, fill=(255, 220, 75, 255))
    draw.text(((bw - (b2[2] - b2[0])) // 2 + text_offset_x // 2, 55), bot_text, font=f_bot, fill=(255, 255, 255, 255))

    bx = (W - bw) // 2
    return ImageClip(np.array(img)).set_start(start_t).set_duration(duration).set_position((bx, 88))

def make_punch_caption(line1, line2, start_t, duration):
    """Renders 2-line tutorial captions below the phone with guaranteed border padding."""
    canvas_w, canvas_h = 980, 210
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font1 = fit_font(draw, line1, canvas_w - 120, 33)
    font2 = fit_font(draw, line2, canvas_w - 120, 38)

    b1 = draw.textbbox((0, 0), line1, font=font1)
    b2 = draw.textbbox((0, 0), line2, font=font2)
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]

    box_w = min(canvas_w - 20, max(w1, w2) + 96)
    box_h = h1 + h2 + 52
    bx = (canvas_w - box_w) // 2
    by = (canvas_h - box_h) // 2

    draw.rounded_rectangle([bx, by, bx + box_w, by + box_h], radius=26, fill=(4, 18, 22, 246), outline=(40, 235, 165, 255), width=4)

    x1 = (canvas_w - w1) // 2
    y1 = by + 16
    draw.text((x1 + 2, y1 + 2), line1, font=font1, fill=(0, 0, 0, 255))
    draw.text((x1, y1), line1, font=font1, fill=(235, 255, 248, 255))

    x2 = (canvas_w - w2) // 2
    y2 = y1 + h1 + 12
    draw.text((x2 + 3, y2 + 3), line2, font=font2, fill=(0, 0, 0, 255))
    draw.text((x2, y2), line2, font=font2, fill=(255, 225, 55, 255))

    return ImageClip(np.array(img)).set_start(start_t).set_duration(duration).set_position(((W - canvas_w) // 2, 1485))

def make_top_progress_bar(total_duration):
    bar_h = 12
    def make_frame(t):
        frame = np.zeros((bar_h, W, 3), dtype=np.uint8)
        frame[:, :] = [8, 24, 28]
        fill_w = int(np.clip(t / total_duration, 0.0, 1.0) * W)
        if fill_w > 0:
            x_grad = np.linspace(0, 1, fill_w)
            frame[:, :fill_w, 0] = (40 + x_grad * 215).astype(np.uint8)
            frame[:, :fill_w, 1] = (235 - x_grad * 15).astype(np.uint8)
            frame[:, :fill_w, 2] = (165 - x_grad * 110).astype(np.uint8)
        return frame
    return VideoClip(make_frame, duration=total_duration).set_fps(30).set_position((0, 0))

def humanize_voice_for_review(voice_path):
    """Adds warm smartphone mic EQ, room reflection, and continuous indoor room tone (-38dB)."""
    raw_wav = "voice_clean.wav"
    human_wav = "voice_human.wav"

    subprocess.run([
        "ffmpeg", "-y", "-i", voice_path,
        "-vn", "-ac", "2", "-ar", "44100",
        "-af", (
            "highpass=f=90,lowpass=f=11200,"
            "equalizer=f=220:t=q:w=1.1:g=1.8,"
            "equalizer=f=2700:t=q:w=1.2:g=1.9,"
            "aecho=0.8:0.88:18:0.07,"
            "acompressor=threshold=-19dB:ratio=2.3:attack=12:release=160,alimiter=limit=0.95"
        ),
        raw_wav
    ], check=True)

    probe = AudioFileClip(raw_wav)
    dur = probe.duration
    probe.close()

    subprocess.run([
        "ffmpeg", "-y",
        "-i", raw_wav,
        "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude=0.0055:duration={dur + 0.2}",
        "-filter_complex",
        f"[0:a]afade=t=out:st={max(0.5, dur - 0.25)}:d=0.25[v];"
        f"[1:a]highpass=f=110,lowpass=f=2200[air];"
        f"[v][air]amix=inputs=2:duration=first:weights=1 0.22",
        human_wav
    ], check=True)

    clip = AudioFileClip(human_wav)
    print(f"🎙️ Humanized Room-Acoustic Voice Ready: {clip.duration:.2f}s")
    return clip

def build_smooth_fintech_bed(duration):
    bed_wav = "bg_pulse.wav"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=146.83:sample_rate=44100:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=220.00:sample_rate=44100:duration={duration}",
        "-filter_complex",
        "[0:a]tremolo=f=1.5:d=0.35,lowpass=f=280,volume=0.07[a0];"
        "[1:a]tremolo=f=3.0:d=0.40,lowpass=f=360,volume=0.04[a1];"
        f"[a0][a1]amix=inputs=2:duration=first,afade=t=in:st=0:d=0.8,afade=t=out:st={duration - 2.3}:d=2.3",
        bed_wav
    ], check=True)
    return AudioFileClip(bed_wav).volumex(0.22)

def build_masterpiece():
    print("🚀 Rendering Full-Starter-Kit FaucetPay 3-Step Coin Swap Tutorial Clip...")
    kit = scan_all_starter_assets()
    screen_path = get_screen_recording()
    voice_path = find_user_voice_file(screen_path)

    logo_dark_path = kit["logos_dark"][0] if kit["logos_dark"] else (kit["logos_light"][0] if kit["logos_light"] else None)
    icon_path = kit["icons"][0] if kit["icons"] else None

    voice = humanize_voice_for_review(voice_path)
    total_dur = max(45.0, round(voice.duration + 2.6, 2))
    print(f"⏱️ Final Tutorial Clip Duration Locked to {total_dur:.2f}s")

    crop_box, raw_dur = get_video_info(screen_path)
    broll_clip = prepare_broll_backdrop_reader(kit["broll"], total_dur)

    intro_dur = 1.5
    outro_dur = 2.3
    body_dur = total_dur - intro_dur - outro_dur

    d1 = body_dur * 0.27
    d2 = body_dur * 0.29
    d3 = body_dur * 0.22
    d4 = body_dur * 0.22

    print("⚡ Pre-trimming 4 fresh Coin Swap tutorial segments via FFmpeg...")
    c_intro = build_official_logo_card(logo_dark_path, icon_path, intro_dur, is_outro=False)
    c_step1_menu = prepare_crystal_phone_clip(screen_path, 1, 10.5, 22.5, raw_dur, crop_box, d1, intro_dur, broll_clip)
    c_step2_pair = prepare_crystal_phone_clip(screen_path, 2, 22.5, 36.5, raw_dur, crop_box, d2, intro_dur + d1, broll_clip)
    c_step2_fee = prepare_crystal_phone_clip(screen_path, 3, 36.5, 49.2, raw_dur, crop_box, d3, intro_dur + d1 + d2, broll_clip)
    c_step3_done = prepare_crystal_phone_clip(screen_path, 4, 49.5, 59.5, raw_dur, crop_box, d4, intro_dur + d1 + d2 + d3, broll_clip)
    c_outro = build_official_logo_card(logo_dark_path, icon_path, outro_dur, is_outro=True)

    base_timeline = concatenate_videoclips(
        [c_intro, c_step1_menu, c_step2_pair, c_step2_fee, c_step3_done, c_outro],
        method="compose"
    ).set_duration(total_dur)

    layers = [base_timeline]

    # Official FaucetPay Watermark throughout the tutorial walkthrough
    if logo_dark_path:
        layers.append(
            ImageClip(logo_dark_path)
            .resize(width=210)
            .set_start(intro_dur)
            .set_duration(body_dur)
            .set_position((W - 255, 24))
        )

    t_s1 = intro_dur
    t_s2 = intro_dur + d1
    t_s3 = intro_dur + d1 + d2 + d3

    # Top Tutorial Step Headers (with official FaucetPay icon if present in starter kit)
    layers.append(make_step_header("STEP 1 OF 3 • OPEN COIN SWAP", "TAP MENU → SELECT COIN SWAP", t_s1, d1, icon_path))
    layers.append(make_step_header("STEP 2 OF 3 • CHOOSE PAIR & RATE", "1 USDT → LTC • ONLY 0.03 USDT FEE", t_s2, d2 + d3, icon_path))
    layers.append(make_step_header("STEP 3 OF 3 • INSTANT EXECUTION", "0.02079889 LTC CREDITED IMMEDIATELY", t_s3, d4, icon_path))

    # Bottom Step-by-Step Tutorial Captions
    captions = [
        ("STEP 1: OPEN BUILT-IN COIN SWAP", "NO EXTERNAL EXCHANGE NEEDED", t_s1, d1),
        ("STEP 2: SELECT COINS & AMOUNT", "CONVERTING 1.00 USDT → LITECOIN", t_s2, d2),
        ("CHECK LIVE RATE & TINY FEE", "ONLY 0.03 USDT (3¢) CONVERSION FEE", t_s2 + d2, d3),
        ("STEP 3: INSTANT WALLET CREDIT!", "0.02079889 LTC • ZERO WAITING TIME", t_s3, d4),
    ]
    for l1, l2, st, dur in captions:
        layers.append(make_punch_caption(l1, l2, st, dur))

    layers.append(make_top_progress_bar(total_dur))

    final_video = CompositeVideoClip(layers, size=(W, H)).set_duration(total_dur)

    # Audio Mix: Humanized Voice + Fintech Bed + All Starter Kit Sonic & SFX Assets
    audio_tracks = []
    audio_tracks.append(voice.set_start(0.35).volumex(1.25))
    audio_tracks.append(build_smooth_fintech_bed(total_dur))

    if kit["sonic"]:
        sonic_file = kit["sonic"][0]
        audio_tracks.append(AudioFileClip(sonic_file).set_start(0.0).volumex(0.48))
        audio_tracks.append(AudioFileClip(sonic_file).set_start(max(0.0, total_dur - outro_dur)).volumex(0.42))

    if kit["sfx_other"]:
        sfx_step = kit["sfx_other"][0]
        audio_tracks.append(AudioFileClip(sfx_step).set_start(t_s2).volumex(0.40))

    if kit["sfx_coin"]:
        audio_tracks.append(AudioFileClip(kit["sfx_coin"][0]).set_start(t_s3 + 0.15).volumex(0.62))

    final_audio = CompositeAudioClip(audio_tracks).set_duration(total_dur)
    final_video = final_video.set_audio(final_audio)

    output_name = "final_faucetpay_shorts.mp4"
    print(f"🎬 Exporting {output_name} ({total_dur:.2f}s Tutorial Clip, 1080x1920, 12Mbps)...")
    final_video.write_videofile(
        output_name,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="12000k",
        threads=2,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )
    print("✨ Full-Starter-Kit FaucetPay Tutorial Clip Rendered Successfully!")

if __name__ == "__main__":
    build_masterpiece()
