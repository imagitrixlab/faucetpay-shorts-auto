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
    ignored = ("final_faucetpay_shorts", "voice_clean", "voice_fitted", "bg_pulse")
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
    """Detects the dark mobile viewport inside the laptop screen capture."""
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
                boxes.append((x + 3, y + 3, x + w - 3, y + h - 3))
    cap.release()
    if boxes:
        med = np.median(np.array(boxes), axis=0).astype(int)
        return tuple(int(v) for v in med)
    return (int(1920 * 0.406), int(1032 * 0.190), int(1920 * 0.594), int(1032 * 0.943))

# Sleek floating 3D titanium phone frame placed in upper-center showcase (leaves y=1565..1920 free for captions!)
PHONE_W, PHONE_H = 730, 1410
PHONE_X = (W - PHONE_W) // 2   # 175
PHONE_Y = 125

def build_studio_stage():
    """Creates a luxury studio fintech stage with 3D drop shadow and glowing titanium phone bezel."""
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.zeros((H, W, 3), dtype=np.float32)
    bg[:, :, 0] = 5 + (y_grid / H) * 7
    bg[:, :, 1] = 10 + (y_grid / H) * 13
    bg[:, :, 2] = 24 + (y_grid / H) * 26

    # Top-center cyan spotlight
    dist_top = np.sqrt(((x_grid - W * 0.5) / 500.0) ** 2 + ((y_grid - 180.0) / 400.0) ** 2)
    glow1 = np.exp(-dist_top ** 2) * 42.0
    bg[:, :, 1] += glow1 * 0.75
    bg[:, :, 2] += glow1 * 1.15

    # Lower caption stage soft ambient accent
    dist_bot = np.sqrt(((x_grid - W * 0.5) / 540.0) ** 2 + ((y_grid - 1700.0) / 260.0) ** 2)
    glow2 = np.exp(-dist_bot ** 2) * 28.0
    bg[:, :, 0] += glow2 * 0.25
    bg[:, :, 1] += glow2 * 0.65
    bg[:, :, 2] += glow2 * 1.05

    pil_bg = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(pil_bg)

    # Outer soft ambient shadow + double metallic/cyan bezel
    draw.rounded_rectangle(
        [PHONE_X - 18, PHONE_Y - 14, PHONE_X + PHONE_W + 18, PHONE_Y + PHONE_H + 22],
        radius=52, fill=(2, 5, 12, 180)
    )
    draw.rounded_rectangle(
        [PHONE_X - 8, PHONE_Y - 8, PHONE_X + PHONE_W + 8, PHONE_Y + PHONE_H + 8],
        radius=44, outline=(0, 215, 255, 215), width=4
    )
    draw.rounded_rectangle(
        [PHONE_X - 3, PHONE_Y - 3, PHONE_X + PHONE_W + 3, PHONE_Y + PHONE_H + 3],
        radius=40, outline=(255, 255, 255, 85), width=2
    )
    return np.array(pil_bg.convert("RGB"))

STUDIO_BG = build_studio_stage()

# Precompute rounded-corner mask for crystal-clear screen edges
_mask_img = Image.new("L", (PHONE_W, PHONE_H), 0)
ImageDraw.Draw(_mask_img).rounded_rectangle([0, 0, PHONE_W - 1, PHONE_H - 1], radius=36, fill=255)
SCREEN_ALPHA = (np.array(_mask_img).astype(np.float32) / 255.0)[:, :, None]

def prepare_crystal_phone_clip(raw_clip, start_t, end_t, crop_box, subtle_zoom=True):
    """Crystal-clear Lanczos4 + Bilateral denoise + fine-radius unsharp mask inside a floating 3D phone."""
    safe_end = min(end_t, max(start_t + 0.5, raw_clip.duration - 0.05))
    sub = raw_clip.subclip(start_t, safe_end).set_fps(30)
    x1, y1, x2, y2 = crop_box
    dur = max(sub.duration, 0.1)

    def render_crystal_frame(get_frame, t):
        frame = get_frame(t)
        cropped = frame[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]

        scale_base = max(PHONE_W / max(cw, 1), PHONE_H / max(ch, 1))
        zoom_factor = 1.0 + (0.03 * min(max(t / dur, 0.0), 1.0) if subtle_zoom else 0.0)
        nw = int(cw * scale_base * zoom_factor)
        nh = int(ch * scale_base * zoom_factor)
        upscaled = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

        cx0 = max(0, (nw - PHONE_W) // 2)
        cy0 = max(0, (nh - PHONE_H) // 2)
        screen = upscaled[cy0:cy0 + PHONE_H, cx0:cx0 + PHONE_W]
        if screen.shape[0] != PHONE_H or screen.shape[1] != PHONE_W:
            screen = cv2.resize(screen, (PHONE_W, PHONE_H), interpolation=cv2.INTER_LANCZOS4)

        # Edge-preserving denoise + fine-radius unsharp mask for razor-sharp UI numbers & text
        denoised = cv2.bilateralFilter(screen, d=3, sigmaColor=22, sigmaSpace=22)
        blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=0.85)
        crisp = cv2.addWeighted(denoised, 1.65, blurred, -0.65, 0)

        # Subtle diagonal glass sheen sweep across the screen
        sweep_x = int(((t / dur) * 1.4 - 0.2) * PHONE_W)
        if 0 <= sweep_x < PHONE_W:
            x_idx = np.arange(PHONE_W)[None, :]
            sheen = np.exp(-((x_idx - sweep_x) / 55.0) ** 2) * 14.0
            crisp = np.clip(crisp.astype(np.float32) + sheen[:, :, None], 0, 255).astype(np.uint8)

        canvas = STUDIO_BG.copy()
        roi = canvas[PHONE_Y:PHONE_Y + PHONE_H, PHONE_X:PHONE_X + PHONE_W].astype(np.float32)
        blended = crisp.astype(np.float32) * SCREEN_ALPHA + roi * (1.0 - SCREEN_ALPHA)
        canvas[PHONE_Y:PHONE_Y + PHONE_H, PHONE_X:PHONE_X + PHONE_W] = blended.astype(np.uint8)
        return canvas

    return sub.fl(render_crystal_frame)

def prepare_native_broll(broll_path, duration, fallback_clip):
    """Loads official 1080x1920 FaucetPay 3D UI B-roll cleanly."""
    if not broll_path:
        return fallback_clip
    clip = VideoFileClip(broll_path).without_audio()
    use_dur = min(duration, max(0.5, clip.duration - 0.05))
    return clip.subclip(0, use_dur).resize((W, H)).set_fps(30)

def smart_key_sticker(video_path, target_width):
    """Keys greenscreen stickers (HSV + spill removal) OR dark-bg stickers (Max-RGB) cleanly."""
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

def make_swap_confirmed_badge(start_t, duration):
    """Renders an authentic, crystal-clear 'SWAP COMPLETED: 1 USDT -> 0.0209 LTC' pill badge."""
    bw, bh = 740, 108
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        f_top = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        f_bot = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except Exception:
        f_top = f_bot = ImageFont.load_default()

    draw.rounded_rectangle([4, 4, bw - 4, bh - 4], radius=50, fill=(6, 40, 28, 242), outline=(34, 235, 130, 255), width=4)
    t1 = "INSTANT SWAP COMPLETED • 0 FEE"
    t2 = "1.00 USDT  →  0.020916 LTC"
    b1 = draw.textbbox((0, 0), t1, font=f_top)
    b2 = draw.textbbox((0, 0), t2, font=f_bot)
    draw.text(((bw - (b1[2] - b1[0])) // 2, 16), t1, font=f_top, fill=(80, 255, 165, 255))
    draw.text(((bw - (b2[2] - b2[0])) // 2, 52), t2, font=f_bot, fill=(255, 255, 255, 255))

    clip = ImageClip(np.array(img)).set_start(start_t).set_duration(duration)
    return clip.set_position(lambda t: ("center", int(185 + max(0.0, (0.14 - t) / 0.14) * -25)))

def make_top_progress_bar(total_duration=30.0):
    """Creates a sleek gold-cyan retention progress bar along the top edge."""
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

def make_punch_caption(line1, line2, start_t, duration, highlight_second=True):
    """Renders 2-line commercial captions in the dedicated lower showcase zone (zero UI overlap)."""
    canvas_w, canvas_h = 940, 225
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 43)
    except Exception:
        font1 = font2 = ImageFont.load_default()

    b1 = draw.textbbox((0, 0), line1, font=font1)
    b2 = draw.textbbox((0, 0), line2, font=font2)
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]

    box_w = min(canvas_w - 16, max(w1, w2) + 74)
    box_h = h1 + h2 + 54
    bx = (canvas_w - box_w) // 2
    by = (canvas_h - box_h) // 2

    draw.rounded_rectangle(
        [bx, by, bx + box_w, by + box_h],
        radius=28,
        fill=(6, 12, 26, 240),
        outline=(0, 215, 255, 255),
        width=4
    )

    x1 = (canvas_w - w1) // 2
    y1 = by + 18
    draw.text((x1 + 2, y1 + 2), line1, font=font1, fill=(0, 0, 0, 255))
    draw.text((x1, y1), line1, font=font1, fill=(235, 245, 255, 255))

    x2 = (canvas_w - w2) // 2
    y2 = y1 + h1 + 12
    c2 = (255, 225, 55, 255) if highlight_second else (0, 225, 255, 255)
    draw.text((x2 + 3, y2 + 3), line2, font=font2, fill=(0, 0, 0, 255))
    draw.text((x2, y2), line2, font=font2, fill=c2)

    base_clip = ImageClip(np.array(img)).set_start(start_t).set_duration(duration)
    # Positioned cleanly at y=1565 below the floating phone frame so 100% of app buttons remain visible
    return base_clip.set_position(lambda t: (35, int(1565 + max(0.0, (0.12 - t) / 0.12) * 24)))

def prepare_studio_voice(voice_path, max_spoken_seconds=27.5):
    """Preserves soft trailing syllables ('portfolio'), masters to -14 LUFS, and fits cleanly inside 27.5s."""
    clean_wav = "voice_clean.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", voice_path,
        "-af", "silenceremove=start_periods=1:start_duration=0.06:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_duration=0.10:start_threshold=-55dB,areverse,apad=pad_dur=0.20,acompressor=threshold=-18dB:ratio=3:attack=10:release=120,loudnorm=I=-14:TP=-1.5:LRA=7",
        "-vn", clean_wav
    ], check=True)

    voice_clip = AudioFileClip(clean_wav)
    print(f"🎙️ Studio-mastered spoken duration: {voice_clip.duration:.2f}s")

    if voice_clip.duration > max_spoken_seconds:
        tempo = min(voice_clip.duration / max_spoken_seconds, 1.35)
        fitted_wav = "voice_fitted.wav"
        print(f"⚡ Applying gentle {tempo:.3f}x tempo fit so 'portfolio' finishes 100% clearly...")
        subprocess.run([
            "ffmpeg", "-y", "-i", clean_wav,
            "-filter:a", f"atempo={tempo:.4f}",
            fitted_wav
        ], check=True)
        voice_clip.close()
        voice_clip = AudioFileClip(fitted_wav)

    return voice_clip

def build_ambient_bed(duration=30.0):
    """Generates a subtle, warm low-pass ambient tech pulse under the voiceover."""
    bed_wav = "bg_pulse.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anoisesrc=color=pink:amplitude=0.015:duration={duration}",
        "-af", "lowpass=f=220,aecho=0.6:0.7:180:0.3,afade=t=in:st=0:d=1.0,afade=t=out:st=27.5:d=2.5",
        bed_wav
    ], check=True)
    return AudioFileClip(bed_wav).volumex(0.32)

def build_masterpiece():
    print("🚀 Starting Ultra-Crystal 100/100 FaucetPay Commercial Short Render...")
    screen_path = find_asset("screen_record*", required=True)
    voice_path = find_asset("voice*", required=True)

    crop_box = detect_mobile_viewport(screen_path)
    print(f"📐 Detected Mobile Viewport Crop Box: {crop_box}")

    raw_screen = VideoFileClip(screen_path).without_audio()

    # 1. Exact Voice-Synced 30.0s Commercial Timeline (Zero BTC mismatch!)
    # [0.0 - 1.5s] Official FaucetPay 3D Brand Intro
    intro_path = find_asset("fp-stream-intro-4s*", required=False)
    if intro_path:
        intro_raw = VideoFileClip(intro_path).without_audio()
        c_intro = intro_raw.subclip(0, min(1.5, intro_raw.duration)).resize((W, H)).set_fps(30)
    else:
        c_intro = prepare_crystal_phone_clip(raw_screen, 0.0, 1.5, crop_box, subtle_zoom=False)

    # [1.5 - 4.5s] Hook: Real $8.62 Live Dashboard
    c_hook_real = prepare_crystal_phone_clip(raw_screen, 0.0, 3.0, crop_box, subtle_zoom=True)

    # [4.5 - 10.5s] Step 1: Official 3D Claim B-roll (3.0s) + Real Verified Faucets & PTC Ads (3.0s)
    c_broll_claim = prepare_native_broll(
        find_asset("fp-broll-ui-claim*", required=False), 3.0,
        prepare_crystal_phone_clip(raw_screen, 67.0, 70.0, crop_box, subtle_zoom=True)
    )
    c_real_ptc = prepare_crystal_phone_clip(raw_screen, 70.0, 73.0, crop_box, subtle_zoom=True)

    # [10.5 - 16.5s] Step 2: Official 3D Balance B-roll (2.5s) + Real 20+ Coins Wallet (3.5s)
    c_broll_bal = prepare_native_broll(
        find_asset("fp-broll-ui-balance*", required=False), 2.5,
        prepare_crystal_phone_clip(raw_screen, 13.0, 15.5, crop_box, subtle_zoom=True)
    )
    c_real_wallet = prepare_crystal_phone_clip(raw_screen, 14.5, 18.0, crop_box, subtle_zoom=True)

    # [16.5 - 23.5s] Step 3: 100% Pure USDT -> LTC Live Swap & Green Confirmation Banner (NO DOGE->BTC clip!)
    c_real_swap = prepare_crystal_phone_clip(raw_screen, 53.5, 60.5, crop_box, subtle_zoom=True)

    # [23.5 - 26.0s] Real Dashboard Trust Close + [26.0 - 30.0s] Official FaucetPay 9:16 Outro
    c_close_real = prepare_crystal_phone_clip(raw_screen, 0.5, 3.0, crop_box, subtle_zoom=False)
    outro_raw = VideoFileClip(find_asset("fp-stream-outro-5s*", required=True)).without_audio()
    c_outro = outro_raw.subclip(0, min(4.0, outro_raw.duration)).resize((W, H)).set_fps(30)

    base_timeline = concatenate_videoclips(
        [c_intro, c_hook_real, c_broll_claim, c_real_ptc, c_broll_bal, c_real_wallet, c_real_swap, c_close_real, c_outro],
        method="compose"
    ).set_duration(30.0)

    layers = [base_timeline]

    # 2. Official Brand Badge & Accurate Graphics (Removed misleading BTC claim pop-up during USDT->LTC swap!)
    logo_pop_path = find_asset("fp-edit-sticker-logo-pop*", required=False)
    if logo_pop_path:
        layers.append(
            smart_key_sticker(logo_pop_path, 380)
            .set_start(1.6)
            .set_duration(2.42)
            .set_position(("center", 200))
        )

    logo_dark_path = find_asset("fp-logo-full-on-dark*", required=False)
    if logo_dark_path:
        layers.append(
            ImageClip(logo_dark_path)
            .resize(width=225)
            .set_start(4.1)
            .set_duration(21.9)
            .set_position((W - 270, 42))
        )

    # Spinning FaucetPay mark during Faucets & Ads (7.5s)
    mark_spin_path = find_asset("fp-edit-sticker-mark-spin*", required=False)
    if mark_spin_path:
        layers.append(
            smart_key_sticker(mark_spin_path, 200)
            .set_start(7.5)
            .set_duration(2.42)
            .set_position((45, 220))
        )

    # Custom Glass Pill Badge confirming exact '1.00 USDT -> 0.020916 LTC' swap (19.0s - 23.2s)
    layers.append(make_swap_confirmed_badge(19.0, 4.2))

    # 3. Commercial 2-Line Captions cleanly below the phone frame (y=1565)
    captions = [
        ("STOP LOSING CRYPTO TO GAS FEES", "ONE MICRO-WALLET FIXES THIS", 0.35, 4.0),
        ("STEP 1: COLLECT FREE CRYPTO", "600+ VERIFIED FAUCETS & ADS", 4.55, 5.7),
        ("STEP 2: STORE EVERYTHING", "HOLD 20+ COINS IN ONE WALLET", 10.55, 5.7),
        ("STEP 3: LIVE INSTANT SWAP", "USDT → LTC WITH ZERO FEES", 16.55, 6.7),
        ("BUILD YOUR PORTFOLIO SAFELY", "START FREE — LINK IN DESCRIPTION", 23.45, 2.75),
    ]
    for l1, l2, st, dur in captions:
        layers.append(make_punch_caption(l1, l2, st, dur))

    # 4. Top Retention Progress Bar
    layers.append(make_top_progress_bar(30.0))

    final_video = CompositeVideoClip(layers, size=(W, H)).set_duration(30.0)

    # 5. Broadcast Studio Audio Mix (Full unclipped 'portfolio' ending + Ambient Bed + Official SFX)
    audio_tracks = []
    voice = prepare_studio_voice(voice_path, max_spoken_seconds=27.5)
    audio_tracks.append(voice.set_start(0.20).volumex(1.15))
    audio_tracks.append(build_ambient_bed(30.0))

    sonic_path = find_asset("fp-audio-sonic-logo*", required=False)
    if sonic_path:
        audio_tracks.append(AudioFileClip(sonic_path).set_start(0.0).volumex(0.50))

    drip_path = find_asset("fp-audio-sfx-coin-drip*", required=False, exclude_word="cascade")
    if drip_path:
        audio_tracks.append(AudioFileClip(drip_path).set_start(7.5).volumex(0.60))

    cascade_path = find_asset("fp-audio-sfx-coin-drip-cascade*", required=False)
    if cascade_path:
        audio_tracks.append(AudioFileClip(cascade_path).set_start(19.0).volumex(0.68))

    final_audio = CompositeAudioClip(audio_tracks).set_duration(30.0)
    final_video = final_video.set_audio(final_audio)

    output_name = "final_faucetpay_shorts.mp4"
    print(f"🎬 Exporting {output_name} (1080x1920, 30fps, 12Mbps Ultra-Crystal Master)...")
    final_video.write_videofile(
        output_name,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="slow",
        bitrate="12000k",
        threads=4,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )
    print("✨ Ultra-Crystal 100/100 Short Rendered Successfully!")

if __name__ == "__main__":
    build_masterpiece()
