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

def find_asset(pattern, required=True, exclude_word=None):
    """Finds assets across repo root or assets/ folder safely (ignores .bin/.onnx)."""
    search_paths = [pattern, os.path.join("assets", pattern), os.path.join("**", pattern)]
    ignored_prefixes = ("final_faucetpay_shorts", "voice_clean", "voice_human", "bg_pulse", "room_air", "tts_", "seg_")
    ignored_exts = (".bin", ".onnx", ".py", ".yml", ".yaml", ".md", ".txt", ".json")

    for sp in search_paths:
        matches = []
        for m in glob.glob(sp, recursive=True):
            fname = os.path.basename(m)
            low = fname.lower()
            if not os.path.isfile(m) or low.startswith(ignored_prefixes) or low.endswith(ignored_exts):
                continue
            if exclude_word and exclude_word in low:
                continue
            matches.append(m)
        if matches:
            print(f"✅ Found [{pattern}] -> {matches[0]}")
            return matches[0]
    if required:
        raise FileNotFoundError(f"❌ Required asset not found matching: {pattern}")
    return None

def get_screen_recording():
    """Downloads the screen recording from Google Drive open link cleanly."""
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
        found = find_asset(pat, required=False)
        if found:
            return found
    raise FileNotFoundError("❌ Screen recording not found!")

def find_user_voice_file(screen_path):
    """Automatically selects the user's uploaded voice audio file in root or assets/."""
    screen_base = os.path.basename(screen_path).lower()
    excluded_prefixes = (
        "final_faucetpay_shorts", "voice_clean", "voice_human", "bg_pulse",
        "room_air", "tts_", "fp-", "60317", "screen_record", "screen_recording", "seg_"
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
    """Gets dimensions, clean crop box, and duration via OpenCV (avoids VFR seek lag)."""
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

# Enlarged Crystal Viewport: Phone at y=75..1445, Captions at y=1472..1685
PHONE_W, PHONE_H = 724, 1370
PHONE_X = (W - PHONE_W) // 2
PHONE_Y = 75

def build_studio_stage():
    """Builds a dark navy FaucetPay-branded studio stage with a titanium bezel."""
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.zeros((H, W, 3), dtype=np.float32)
    bg[:, :, 0] = 6 + (y_grid / H) * 8
    bg[:, :, 1] = 12 + (y_grid / H) * 14
    bg[:, :, 2] = 28 + (y_grid / H) * 28

    dist_top = np.sqrt(((x_grid - W * 0.5) / 520.0) ** 2 + ((y_grid - 160.0) / 380.0) ** 2)
    glow1 = np.exp(-dist_top ** 2) * 44.0
    bg[:, :, 1] += glow1 * 0.75
    bg[:, :, 2] += glow1 * 1.15

    pil_bg = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(pil_bg)

    draw.rounded_rectangle(
        [PHONE_X - 16, PHONE_Y - 14, PHONE_X + PHONE_W + 16, PHONE_Y + PHONE_H + 18],
        radius=48, fill=(2, 5, 12, 195)
    )
    draw.rounded_rectangle(
        [PHONE_X - 7, PHONE_Y - 7, PHONE_X + PHONE_W + 7, PHONE_Y + PHONE_H + 7],
        radius=40, outline=(0, 215, 255, 230), width=4
    )
    return np.array(pil_bg.convert("RGB"))

STUDIO_BG = build_studio_stage()

_mask_img = Image.new("L", (PHONE_W, PHONE_H), 0)
ImageDraw.Draw(_mask_img).rounded_rectangle([0, 0, PHONE_W - 1, PHONE_H - 1], radius=34, fill=255)
SCREEN_ALPHA = (np.array(_mask_img).astype(np.float32) / 255.0)[:, :, None]

def prepare_crystal_phone_clip(screen_path, seg_idx, start_t, end_t, raw_dur, crop_box, target_dur):
    """Pre-trims & converts each segment to 30fps CFR via native FFmpeg so rendering takes ~90 seconds."""
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
        f"unsharp=3:3:0.45:3:3:0.0,"
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
        canvas = STUDIO_BG.copy()
        roi = canvas[PHONE_Y:PHONE_Y + PHONE_H, PHONE_X:PHONE_X + PHONE_W].astype(np.float32)
        blended = screen.astype(np.float32) * SCREEN_ALPHA + roi * (1.0 - SCREEN_ALPHA)
        canvas[PHONE_Y:PHONE_Y + PHONE_H, PHONE_X:PHONE_X + PHONE_W] = blended.astype(np.uint8)
        return canvas

    return sub.fl(composite_on_stage).set_duration(target_dur)

def build_official_logo_card(logo_path, duration, is_outro=False):
    """Creates a 100% original animated card using FaucetPay's official PNG logo (0% stock video hash)."""
    bg = STUDIO_BG.copy()
    pil_card = Image.fromarray(bg).convert("RGBA")
    draw = ImageDraw.Draw(pil_card)

    draw.rounded_rectangle([85, 520, W - 85, 1380], radius=44, fill=(8, 18, 40, 245), outline=(0, 215, 255, 255), width=5)

    if logo_path and os.path.exists(logo_path):
        logo_img = Image.open(logo_path).convert("RGBA")
        lw = 640
        lh = int(logo_img.height * (lw / logo_img.width))
        logo_img = logo_img.resize((lw, lh), Image.LANCZOS)
        pil_card.paste(logo_img, ((W - lw) // 2, 650), logo_img)

    try:
        f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 43)
        f_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 33)
    except Exception:
        f_title = f_sub = ImageFont.load_default()

    if is_outro:
        line1 = "TRY FAUCETPAY MICRO-WALLET"
        line2 = "OFFICIAL LINK IN DESCRIPTION ↓"
    else:
        line1 = "MY REAL FAUCETPAY WALKTHROUGH"
        line2 = "LIVE BALANCE • INSTANT SWAP • FAUCETS & PTC"

    b1 = draw.textbbox((0, 0), line1, font=f_title)
    b2 = draw.textbbox((0, 0), line2, font=f_sub)
    draw.text(((W - (b1[2] - b1[0])) // 2, 960), line1, font=f_title, fill=(255, 255, 255, 255))
    draw.text(((W - (b2[2] - b2[0])) // 2, 1045), line2, font=f_sub, fill=(255, 225, 55, 255))

    card_rgb = np.array(pil_card.convert("RGB"))
    return ImageClip(card_rgb).set_duration(duration).set_fps(30)

def make_status_pill(top_text, bot_text, start_t, duration):
    """Renders a crystal-clear live verification badge at the top of the screen."""
    bw, bh = 790, 106
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        f_top = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        f_bot = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    except Exception:
        f_top = f_bot = ImageFont.load_default()

    draw.rounded_rectangle([4, 4, bw - 4, bh - 4], radius=50, fill=(6, 40, 28, 242), outline=(34, 235, 130, 255), width=4)
    b1 = draw.textbbox((0, 0), top_text, font=f_top)
    b2 = draw.textbbox((0, 0), bot_text, font=f_bot)
    draw.text(((bw - (b1[2] - b1[0])) // 2, 16), top_text, font=f_top, fill=(80, 255, 165, 255))
    draw.text(((bw - (b2[2] - b2[0])) // 2, 52), bot_text, font=f_bot, fill=(255, 255, 255, 255))

    bx = (W - bw) // 2
    return ImageClip(np.array(img)).set_start(start_t).set_duration(duration).set_position((bx, 125))

def make_punch_caption(line1, line2, start_t, duration):
    """Renders 2-line review captions in the bottom safe zone (y=1472..1685)."""
    canvas_w, canvas_h = 940, 215
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception:
        font1 = font2 = ImageFont.load_default()

    b1 = draw.textbbox((0, 0), line1, font=font1)
    b2 = draw.textbbox((0, 0), line2, font=font2)
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]

    box_w = min(canvas_w - 16, max(w1, w2) + 72)
    box_h = h1 + h2 + 50
    bx = (canvas_w - box_w) // 2
    by = (canvas_h - box_h) // 2

    draw.rounded_rectangle([bx, by, bx + box_w, by + box_h], radius=26, fill=(6, 12, 26, 242), outline=(0, 215, 255, 255), width=4)

    x1 = (canvas_w - w1) // 2
    y1 = by + 16
    draw.text((x1 + 2, y1 + 2), line1, font=font1, fill=(0, 0, 0, 255))
    draw.text((x1, y1), line1, font=font1, fill=(235, 245, 255, 255))

    x2 = (canvas_w - w2) // 2
    y2 = y1 + h1 + 11
    draw.text((x2 + 3, y2 + 3), line2, font=font2, fill=(0, 0, 0, 255))
    draw.text((x2, y2), line2, font=font2, fill=(255, 225, 55, 255))

    return ImageClip(np.array(img)).set_start(start_t).set_duration(duration).set_position((35, 1472))

def make_top_progress_bar(total_duration):
    bar_h = 12
    def make_frame(t):
        frame = np.zeros((bar_h, W, 3), dtype=np.uint8)
        frame[:, :] = [12, 18, 36]
        fill_w = int(np.clip(t / total_duration, 0.0, 1.0) * W)
        if fill_w > 0:
            x_grad = np.linspace(0, 1, fill_w)
            frame[:, :fill_w, 0] = (x_grad * 255).astype(np.uint8)
            frame[:, :fill_w, 1] = (210 + x_grad * 20).astype(np.uint8)
            frame[:, :fill_w, 2] = (255 - x_grad * 195).astype(np.uint8)
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
        "-f", "lavfi", "-i", f"sine=frequency=130.81:sample_rate=44100:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=196.00:sample_rate=44100:duration={duration}",
        "-filter_complex",
        "[0:a]tremolo=f=1.5:d=0.35,lowpass=f=280,volume=0.07[a0];"
        "[1:a]tremolo=f=3.0:d=0.40,lowpass=f=360,volume=0.04[a1];"
        f"[a0][a1]amix=inputs=2:duration=first,afade=t=in:st=0:d=0.8,afade=t=out:st={duration - 2.5}:d=2.5",
        bed_wav
    ], check=True)
    return AudioFileClip(bed_wav).volumex(0.22)

def build_masterpiece():
    print("🚀 Rendering Crystal-Clear 85+ Score FaucetPay Full Review Short...")
    screen_path = get_screen_recording()
    voice_path = find_user_voice_file(screen_path)
    logo_dark_path = find_asset("fp-logo-full-on-dark*", required=False)

    voice = humanize_voice_for_review(voice_path)
    total_dur = max(53.0, round(voice.duration + 2.8, 2))
    print(f"⏱️ Final Video Duration Locked to {total_dur:.2f}s")

    crop_box, raw_dur = get_video_info(screen_path)

    intro_dur = 1.5
    outro_dur = 2.5
    body_dur = total_dur - intro_dur - outro_dur

    print("⚡ Pre-trimming 6 CFR 30fps crystal segments via FFmpeg...")
    c_intro = build_official_logo_card(logo_dark_path, intro_dur, is_outro=False)
    c_wallet = prepare_crystal_phone_clip(screen_path, 1, 0.0, 6.0, raw_dur, crop_box, body_dur * 0.16)
    c_withdraw = prepare_crystal_phone_clip(screen_path, 2, 6.0, 10.2, raw_dur, crop_box, body_dur * 0.12)
    c_swap_setup = prepare_crystal_phone_clip(screen_path, 3, 34.0, 49.2, raw_dur, crop_box, body_dur * 0.26)
    c_swap_done = prepare_crystal_phone_clip(screen_path, 4, 49.5, 54.5, raw_dur, crop_box, body_dur * 0.14)
    c_faucets = prepare_crystal_phone_clip(screen_path, 5, 62.0, 68.5, raw_dur, crop_box, body_dur * 0.16)
    c_ptc = prepare_crystal_phone_clip(screen_path, 6, 74.0, 82.5, raw_dur, crop_box, body_dur * 0.16)
    c_outro = build_official_logo_card(logo_dark_path, outro_dur, is_outro=True)

    base_timeline = concatenate_videoclips(
        [c_intro, c_wallet, c_withdraw, c_swap_setup, c_swap_done, c_faucets, c_ptc, c_outro],
        method="compose"
    ).set_duration(total_dur)

    layers = [base_timeline]

    if logo_dark_path:
        layers.append(
            ImageClip(logo_dark_path)
            .resize(width=220)
            .set_start(intro_dur)
            .set_duration(body_dur)
            .set_position((W - 265, 28))
        )

    t_swap_done = intro_dur + body_dur * 0.54
    t_earn = intro_dur + body_dur * 0.68

    layers.append(make_status_pill("MY LIVE FAUCETPAY WALLET", "BALANCE: $6.03 • 23 COINS + WITHDRAWALS", 2.0, body_dur * 0.26))
    layers.append(make_status_pill("INSTANT SWAP COMPLETED • 0.03 FEE", "1.00 USDT  →  0.02079889 LTC", t_swap_done, body_dur * 0.14))
    layers.append(make_status_pill("600+ FAUCET SITES & PTC LISTINGS", "INSTANT DIRECT PAYOUTS • NO DEPOSIT FEE", t_earn, body_dur * 0.30))

    captions = [
        ("MY REAL FAUCETPAY EXPERIENCE", "$6.03 ACROSS 23 COINS + WITHDRAWALS", intro_dur, body_dur * 0.28),
        ("BUILT-IN COIN SWAP TOOL", "1 USDT → LITECOIN • 3¢ MINIMAL FEE", intro_dur + body_dur * 0.28, body_dur * 0.26),
        ("SWAP COMPLETED INSTANTLY!", "0.02079889 LTC CREDITED LIVE", t_swap_done, body_dur * 0.14),
        ("FAUCET SITES & PTC LISTINGS", "DIRECT INSTANT PAYOUTS • NO DEPOSIT FEE", t_earn, body_dur * 0.32),
    ]
    for l1, l2, st, dur in captions:
        layers.append(make_punch_caption(l1, l2, st, dur))

    layers.append(make_top_progress_bar(total_dur))

    final_video = CompositeVideoClip(layers, size=(W, H)).set_duration(total_dur)

    audio_tracks = []
    audio_tracks.append(voice.set_start(0.35).volumex(1.25))
    audio_tracks.append(build_smooth_fintech_bed(total_dur))

    sonic_path = find_asset("fp-audio-sonic-logo*", required=False)
    if sonic_path:
        audio_tracks.append(AudioFileClip(sonic_path).set_start(0.0).volumex(0.45))

    cascade_path = find_asset("fp-audio-sfx-coin-drip-cascade*", required=False)
    if cascade_path:
        audio_tracks.append(AudioFileClip(cascade_path).set_start(t_swap_done + 0.3).volumex(0.60))

    final_audio = CompositeAudioClip(audio_tracks).set_duration(total_dur)
    final_video = final_video.set_audio(final_audio)

    output_name = "final_faucetpay_shorts.mp4"
    print(f"🎬 Exporting {output_name} ({total_dur:.2f}s Full Review, 1080x1920, 12Mbps)...")
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
    print("✨ High-Score (85+) FaucetPay Review Short Rendered Successfully!")

if __name__ == "__main__":
    build_masterpiece()
