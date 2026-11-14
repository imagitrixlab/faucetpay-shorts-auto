import os
import glob
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Pillow 10+ compatibility fix for MoviePy 1.0.3
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
    CompositeAudioClip, concatenate_videoclips
)
from moviepy.video.fx.all import mask_color, resize

W, H = 1080, 1920

def find_asset(pattern):
    """Searches repo root, assets/ folder, and any subfolder so separate uploads always work."""
    search_paths = [
        pattern,
        os.path.join("assets", pattern),
        os.path.join("**", pattern),
    ]
    ignored = ("final_faucetpay_shorts", "voice_clean", "voice_fitted")
    for sp in search_paths:
        matches = [
            m for m in glob.glob(sp, recursive=True)
            if os.path.isfile(m) and not os.path.basename(m).startswith(ignored)
        ]
        if matches:
            print(f"✅ Found [{pattern}] -> {matches[0]}")
            return matches[0]
    raise FileNotFoundError(f"❌ File not found anywhere in repo matching: {pattern}")

def detect_mobile_viewport(video_path):
    """Detects the 412x924 dark mobile viewport inside the laptop screen capture."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    fh, fw, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 65).astype(np.uint8)
    dark_mask[:, :int(fw * 0.30)] = 0
    dark_mask[:, int(fw * 0.70):] = 0
    dark_mask[:int(fh * 0.14), :] = 0
    dark_mask[int(fh * 0.96):, :] = 0

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if w > 200 and h > 400:
            return (x + 2, y + 2, x + w - 2, y + h - 2)
    return (int(fw * 0.406), int(fh * 0.190), int(fw * 0.594), int(fh * 0.943))

def prepare_screen_clip(raw_clip, start_t, end_t, crop_box, subtle_zoom=True):
    safe_end = min(end_t, max(start_t + 0.5, raw_clip.duration - 0.1))
    sub = raw_clip.subclip(start_t, safe_end)
    x1, y1, x2, y2 = crop_box
    cropped = sub.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    scaled = cropped.resize(width=W)
    if scaled.h < H:
        scaled = cropped.resize(height=H)
    base = scaled.crop(x_center=scaled.w / 2, y_center=scaled.h / 2, width=W, height=H)
    if subtle_zoom:
        dur = base.duration
        base = base.resize(lambda t: 1.0 + 0.04 * (t / max(dur, 0.1))).crop(
            x_center=W / 2, y_center=H / 2, width=W, height=H
        )
    return base

def make_caption_clip(text, start_t, duration):
    """Renders broadcast-grade kinetic captions positioned safely above Shorts UI."""
    img = Image.new("RGBA", (W, 240), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 42, 22
    box_w, box_h = tw + pad_x * 2, th + pad_y * 2
    bx = (W - box_w) // 2
    by = (240 - box_h) // 2

    draw.rounded_rectangle(
        [bx, by, bx + box_w, by + box_h],
        radius=26,
        fill=(10, 16, 30, 225),
        outline=(0, 210, 255, 255),
        width=4
    )
    tx = (W - tw) // 2
    ty = by + pad_y - 4
    draw.text((tx + 3, ty + 3), text, font=font, fill=(0, 0, 0, 255))
    draw.text((tx, ty), text, font=font, fill=(255, 230, 60, 255))

    return (ImageClip(np.array(img))
            .set_start(start_t)
            .set_duration(duration)
            .set_position(("center", 1450)))

def prepare_zero_cutoff_voice(voice_path, max_spoken_seconds=29.0):
    """Strips trailing silence and gently fits full speech so zero words are ever cut off."""
    clean_wav = "voice_clean.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", voice_path,
        "-af", "areverse,silenceremove=start_periods=1:start_duration=0.15:start_threshold=-42dB,areverse",
        "-vn", clean_wav
    ], check=True)

    voice_clip = AudioFileClip(clean_wav)
    print(f"🎙️ Detected pure spoken duration: {voice_clip.duration:.2f}s")

    if voice_clip.duration > max_spoken_seconds:
        tempo = min(voice_clip.duration / max_spoken_seconds, 1.35)
        fitted_wav = "voice_fitted.wav"
        print(f"⚡ Applying gentle {tempo:.3f}x tempo fit so zero syllables are cut...")
        subprocess.run([
            "ffmpeg", "-y", "-i", clean_wav,
            "-filter:a", f"atempo={tempo:.4f}",
            fitted_wav
        ], check=True)
        voice_clip.close()
        voice_clip = AudioFileClip(fitted_wav)

    return voice_clip

def build_masterpiece():
    print("🚀 Starting World-Class FaucetPay Short Generation...")
    screen_path = find_asset("screen_record*")
    voice_path = find_asset("voice*")

    crop_box = detect_mobile_viewport(screen_path)
    print(f"📐 Detected Mobile Viewport Crop Box: {crop_box}")

    raw_screen = VideoFileClip(screen_path).without_audio()

    # 1. Best Real-Account Proof Segments (Total = 26.0s)
    clip_hook_dash = prepare_screen_clip(raw_screen, 0.0, 3.0, crop_box, subtle_zoom=False)
    clip_wallet    = prepare_screen_clip(raw_screen, 14.0, 20.0, crop_box, subtle_zoom=True)
    clip_swap_live = prepare_screen_clip(raw_screen, 51.5, 62.0, crop_box, subtle_zoom=True)
    clip_earn_ptc  = prepare_screen_clip(raw_screen, 69.5, 76.0, crop_box, subtle_zoom=True)

    # 2. Official FaucetPay Outro (4.0s at end -> Total 30.0s)
    outro_raw = VideoFileClip(find_asset("fp-stream-outro-5s*")).without_audio()
    clip_outro = outro_raw.subclip(0, min(4.0, outro_raw.duration)).resize((W, H))

    base_timeline = concatenate_videoclips(
        [clip_hook_dash, clip_wallet, clip_swap_live, clip_earn_ptc, clip_outro],
        method="compose"
    ).set_duration(30.0)

    layers = [base_timeline]

    # 3. Official FaucetPay Brand Badge & Green-Screen Stickers
    logo_badge = (ImageClip(find_asset("fp-logo-full-on-dark*"))
                  .resize(width=240)
                  .set_start(2.5)
                  .set_duration(23.5)
                  .set_position((W - 280, 70)))
    layers.append(logo_badge)

    sticker_claim = (VideoFileClip(find_asset("fp-edit-sticker-claim-pop*"))
                     .without_audio()
                     .resize(width=450)
                     .fx(mask_color, color=[0, 255, 0], thr=115, s=12)
                     .set_start(14.5)
                     .set_duration(2.5)
                     .set_position(("center", 300)))
    layers.append(sticker_claim)

    sticker_spin = (VideoFileClip(find_asset("fp-edit-sticker-mark-spin*"))
                    .without_audio()
                    .resize(width=230)
                    .fx(mask_color, color=[0, 255, 0], thr=115, s=12)
                    .set_start(20.5)
                    .set_duration(2.5)
                    .set_position((60, 280)))
    layers.append(sticker_spin)

    # 4. Kinetic Subtitles
    captions = [
        ("ONE WALLET. 20+ CRYPTOS.", 0.3, 3.2),
        ("ZERO GAS FEE MICRO-WALLET", 3.6, 4.4),
        ("600+ VERIFIED FAUCETS & PTC", 8.2, 4.3),
        ("LIVE INSTANT SWAP: USDT → LTC", 12.7, 4.5),
        ("COMPLETED IN SECONDS! ⚡", 17.3, 3.7),
        ("START FREE — LINK IN BIO 👇", 21.2, 4.8),
    ]
    for text, st, dur in captions:
        layers.append(make_caption_clip(text, st, dur))

    final_video = CompositeVideoClip(layers, size=(W, H)).set_duration(30.0)

    # 5. Zero-Cutoff Studio Audio Mix
    audio_tracks = []
    voice = prepare_zero_cutoff_voice(voice_path, max_spoken_seconds=29.0)
    audio_tracks.append(voice.set_start(0.3).volumex(1.18))

    sfx_sonic = AudioFileClip(find_asset("fp-audio-sonic-logo*")).set_start(0.0).volumex(0.45)
    sfx_cascade = AudioFileClip(find_asset("fp-audio-sfx-coin-drip-cascade*")).set_start(14.5).volumex(0.65)
    sfx_drip = AudioFileClip(find_asset("fp-audio-sfx-coin-drip.*")).set_start(20.5).volumex(0.60)
    audio_tracks.extend([sfx_sonic, sfx_cascade, sfx_drip])

    final_audio = CompositeAudioClip(audio_tracks).set_duration(30.0)
    final_video = final_video.set_audio(final_audio)

    output_name = "final_faucetpay_shorts.mp4"
    print(f"🎬 Exporting {output_name} (1080x1920, 30fps)...")
    final_video.write_videofile(
        output_name,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="6000k",
        threads=4
    )
    print("✨ World-Class Short Rendered Successfully!")

if __name__ == "__main__":
    build_masterpiece()
