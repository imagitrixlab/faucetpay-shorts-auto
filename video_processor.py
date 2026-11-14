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
    VideoFileClip, AudioFileClip, ImageClip, VideoClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips
)

W, H = 1080, 1920

def find_asset(pattern, required=True, exclude_word=None):
    """Finds assets across repo root, assets/ folder, or any subfolder safely."""
    search_paths = [
        pattern,
        os.path.join("assets", pattern),
        os.path.join("**", pattern),
    ]
    ignored = ("final_faucetpay_shorts", "voice_clean", "voice_fitted")
    for sp in search_paths:
        matches = []
        for m in glob.glob(sp, recursive=True):
            fname = os.path.basename(m)
            if not os.path.isfile(m) or fname.startswith(ignored):
                continue
            if exclude_word and exclude_word in fname:
                continue
            matches.append(m)
        if matches:
            print(f"✅ Found [{pattern}] -> {matches[0]}")
            return matches[0]
    if required:
        raise FileNotFoundError(f"❌ Required asset not found matching: {pattern}")
    print(f"⚠️ Optional asset skipped (not found): {pattern}")
    return None

def detect_mobile_viewport(video_path):
    """Detects the 412x924 dark mobile viewport across multiple frames for rock-solid accuracy."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    boxes = []
    for ratio in (0.01, 0.15, 0.40):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * ratio))
        ret, frame = cap.read()
        if not ret:
            continue
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
                boxes.append((x + 2, y + 2, x + w - 2, y + h - 2))
    cap.release()
    if boxes:
        med = np.median(np.array(boxes), axis=0).astype(int)
        return tuple(int(v) for v in med)
    return (int(1920 * 0.406), int(1032 * 0.190), int(1920 * 0.594), int(1032 * 0.943))

def prepare_screen_clip(raw_clip, start_t, end_t, crop_box, subtle_zoom=True):
    """Crops mobile viewport cleanly to 1080x1920 with zero-drift OpenCV center zoom."""
    safe_end = min(end_t, max(start_t + 0.5, raw_clip.duration - 0.05))
    sub = raw_clip.subclip(start_t, safe_end)
    x1, y1, x2, y2 = crop_box
    cropped = sub.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    scaled = cropped.resize(width=W)
    if scaled.h < H:
        scaled = cropped.resize(height=H)
    base = scaled.crop(x_center=scaled.w / 2, y_center=scaled.h / 2, width=W, height=H).set_fps(30)
    if subtle_zoom:
        dur = max(base.duration, 0.1)
        def zoom_center(get_frame, t):
            frame = get_frame(t)
            scale = 1.0 + 0.045 * min(max(t / dur, 0.0), 1.0)
            nw, nh = int(W * scale), int(H * scale)
            zoomed = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            x0 = (nw - W) // 2
            y0 = (nh - H) // 2
            return zoomed[y0:y0 + H, x0:x0 + W]
        base = base.fl(zoom_center)
    return base

def smart_key_sticker(video_path, target_width):
    """Keys greenscreen stickers (HSV + spill removal) OR dark-bg stickers (Max-RGB) with 100% solid colors."""
    raw = VideoFileClip(video_path).without_audio().resize(width=target_width).set_fps(30)
    is_greenscreen = "greenscreen" in os.path.basename(video_path).lower()
    max_t = max(0.0, raw.duration - 0.04)

    if is_greenscreen:
        def remove_green_spill(frame):
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            spill = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 92) & (hsv[:, :, 1] > 55)
            hsv[spill, 1] *= 0.12
            return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

        def make_green_mask(t):
            frame = raw.get_frame(min(max(0.0, t), max_t))
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            bg = cv2.inRange(hsv, np.array([36, 70, 50]), np.array([88, 255, 255]))
            fg = cv2.GaussianBlur(255 - bg, (5, 5), 0)
            return fg.astype(np.float32) / 255.0

        clean_rgb = raw.fl_image(remove_green_spill)
        mask_clip = VideoClip(make_green_mask, ismask=True, duration=raw.duration).set_fps(30)
        return clean_rgb.set_mask(mask_clip)
    else:
        def make_maxrgb_mask(t):
            frame = raw.get_frame(min(max(0.0, t), max_t))
            peak = np.max(frame, axis=2).astype(np.float32)
            fg = np.clip((peak - 8.0) * (255.0 / 18.0), 0, 255)
            fg = cv2.GaussianBlur(fg, (5, 5), 0)
            return fg / 255.0

        mask_clip = VideoClip(make_maxrgb_mask, ismask=True, duration=raw.duration).set_fps(30)
        return raw.set_mask(mask_clip)

def make_top_progress_bar(total_duration=30.0):
    """Creates a sleek gold-cyan retention progress bar along the top edge."""
    bar_h = 12
    def make_frame(t):
        frame = np.zeros((bar_h, W, 3), dtype=np.uint8)
        frame[:, :] = [15, 23, 42]
        fill_w = int(np.clip(t / total_duration, 0.0, 1.0) * W)
        if fill_w > 0:
            x_grad = np.linspace(0, 1, fill_w)
            frame[:, :fill_w, 0] = (x_grad * 255).astype(np.uint8)
            frame[:, :fill_w, 1] = (210 + x_grad * 20).astype(np.uint8)
            frame[:, :fill_w, 2] = (255 - x_grad * 195).astype(np.uint8)
        return frame
    return VideoClip(make_frame, duration=total_duration).set_fps(30).set_position((0, 0))

def make_caption_clip(text, start_t, duration):
    """Renders broadcast-grade kinetic captions safely inside YouTube Shorts UI bounds."""
    canvas_w = 900
    img = Image.new("RGBA", (canvas_w, 210), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 38, 22
    box_w, box_h = min(canvas_w - 16, tw + pad_x * 2), th + pad_y * 2
    bx = (canvas_w - box_w) // 2
    by = (210 - box_h) // 2

    draw.rounded_rectangle(
        [bx, by, bx + box_w, by + box_h],
        radius=26,
        fill=(8, 14, 28, 235),
        outline=(0, 215, 255, 255),
        width=4
    )
    tx = (canvas_w - tw) // 2
    ty = by + pad_y - 4
    draw.text((tx + 3, ty + 3), text, font=font, fill=(0, 0, 0, 255))
    draw.text((tx, ty), text, font=font, fill=(255, 232, 65, 255))

    return (ImageClip(np.array(img))
            .set_start(start_t)
            .set_duration(duration)
            .set_position((55, 1370)))

def prepare_studio_voice(voice_path, max_spoken_seconds=28.8):
    """Trims start/end silence, masters voice to -14 LUFS broadcast clarity, and prevents any word cutoff."""
    clean_wav = "voice_clean.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", voice_path,
        "-af", "silenceremove=start_periods=1:start_duration=0.08:start_threshold=-42dB,areverse,silenceremove=start_periods=1:start_duration=0.15:start_threshold=-42dB,areverse,acompressor=threshold=-18dB:ratio=3:attack=10:release=120,loudnorm=I=-14:TP=-1.5:LRA=7",
        "-vn", clean_wav
    ], check=True)

    voice_clip = AudioFileClip(clean_wav)
    print(f"🎙️ Studio-mastered spoken duration: {voice_clip.duration:.2f}s")

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
    print("🚀 Starting 100/100 International-Grade FaucetPay Short Render...")
    screen_path = find_asset("screen_record*", required=True)
    voice_path = find_asset("voice*", required=True)

    crop_box = detect_mobile_viewport(screen_path)
    print(f"📐 Detected Mobile Viewport Crop Box: {crop_box}")

    raw_screen = VideoFileClip(screen_path).without_audio()

    # 1. Exact 30.0s International Ad Timeline
    intro_path = find_asset("fp-stream-intro-4s*", required=False)
    if intro_path:
        intro_raw = VideoFileClip(intro_path).without_audio()
        clip_intro = intro_raw.subclip(0, min(1.5, intro_raw.duration)).resize((W, H)).set_fps(30)
    else:
        clip_intro = prepare_screen_clip(raw_screen, 0.0, 1.5, crop_box, subtle_zoom=False)

    clip_hook_dash = prepare_screen_clip(raw_screen, 0.0, 3.0, crop_box, subtle_zoom=False)
    clip_wallet    = prepare_screen_clip(raw_screen, 14.0, 19.0, crop_box, subtle_zoom=True)
    clip_swap_live = prepare_screen_clip(raw_screen, 51.5, 61.5, crop_box, subtle_zoom=True)
    clip_earn_ptc  = prepare_screen_clip(raw_screen, 69.5, 76.0, crop_box, subtle_zoom=True)

    outro_path = find_asset("fp-stream-outro-5s*", required=True)
    outro_raw = VideoFileClip(outro_path).without_audio()
    clip_outro = outro_raw.subclip(0, min(4.0, outro_raw.duration)).resize((W, H)).set_fps(30)

    base_timeline = concatenate_videoclips(
        [clip_intro, clip_hook_dash, clip_wallet, clip_swap_live, clip_earn_ptc, clip_outro],
        method="compose"
    ).set_duration(30.0)

    layers = [base_timeline]

    # 2. Official FaucetPay Stickers & Brand Watermark (2.42s safe boundary duration)
    logo_pop_path = find_asset("fp-edit-sticker-logo-pop*", required=False)
    if logo_pop_path:
        sticker_logo_pop = (smart_key_sticker(logo_pop_path, 420)
                            .set_start(1.6)
                            .set_duration(2.42)
                            .set_position(("center", 250)))
        layers.append(sticker_logo_pop)

    logo_dark_path = find_asset("fp-logo-full-on-dark*", required=False)
    if logo_dark_path:
        logo_badge = (ImageClip(logo_dark_path)
                      .resize(width=240)
                      .set_start(4.1)
                      .set_duration(21.9)
                      .set_position((W - 280, 65)))
        layers.append(logo_badge)

    claim_pop_path = find_asset("fp-edit-sticker-claim-pop*", required=False)
    if claim_pop_path:
        sticker_claim = (smart_key_sticker(claim_pop_path, 440)
                         .set_start(14.5)
                         .set_duration(2.42)
                         .set_position(("center", 280)))
        layers.append(sticker_claim)

    mark_spin_path = find_asset("fp-edit-sticker-mark-spin*", required=False)
    if mark_spin_path:
        sticker_spin = (smart_key_sticker(mark_spin_path, 220)
                        .set_start(20.5)
                        .set_duration(2.42)
                        .set_position((55, 260)))
        layers.append(sticker_spin)

    # 3. Safe-Zone Kinetic Captions (100% glyph-safe, clears at 26.2s before Official Outro)
    captions = [
        ("ONE WALLET. 20+ CRYPTOS.", 0.4, 3.6),
        ("ZERO GAS FEE MICRO-WALLET", 4.2, 4.6),
        ("LIVE INSTANT SWAP: USDT → LTC", 9.2, 5.0),
        ("COMPLETED IN SECONDS!", 14.4, 4.6),
        ("600+ VERIFIED FAUCETS & PTC", 19.3, 3.6),
        ("START FREE — LINK IN DESCRIPTION", 23.1, 3.1),
    ]
    for text, st, dur in captions:
        layers.append(make_caption_clip(text, st, dur))

    # 4. Top Retention Progress Bar
    layers.append(make_top_progress_bar(30.0))

    final_video = CompositeVideoClip(layers, size=(W, H)).set_duration(30.0)

    # 5. Broadcast Audio Mix
    audio_tracks = []
    voice = prepare_studio_voice(voice_path, max_spoken_seconds=28.8)
    audio_tracks.append(voice.set_start(0.35).volumex(1.12))

    sonic_path = find_asset("fp-audio-sonic-logo*", required=False)
    if sonic_path:
        audio_tracks.append(AudioFileClip(sonic_path).set_start(0.0).volumex(0.50))

    cascade_path = find_asset("fp-audio-sfx-coin-drip-cascade*", required=False)
    if cascade_path:
        audio_tracks.append(AudioFileClip(cascade_path).set_start(14.5).volumex(0.65))

    drip_path = find_asset("fp-audio-sfx-coin-drip*", required=False, exclude_word="cascade")
    if drip_path:
        audio_tracks.append(AudioFileClip(drip_path).set_start(20.5).volumex(0.60))

    final_audio = CompositeAudioClip(audio_tracks).set_duration(30.0)
    final_video = final_video.set_audio(final_audio)

    output_name = "final_faucetpay_shorts.mp4"
    print(f"🎬 Exporting {output_name} (1080x1920, 30fps, Broadcast YUV420P)...")
    final_video.write_videofile(
        output_name,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="7000k",
        threads=4,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )
    print("✨ 100/100 International-Grade Short Rendered Successfully!")

if __name__ == "__main__":
    build_masterpiece()
