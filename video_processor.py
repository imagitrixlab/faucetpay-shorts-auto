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

def build_luxury_backdrop():
    """Creates a deep studio fintech backdrop with subtle cyan/gold ambient glow."""
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.zeros((H, W, 3), dtype=np.float32)
    # Deep midnight navy gradient
    bg[:, :, 0] = 6 + (y_grid / H) * 8
    bg[:, :, 1] = 11 + (y_grid / H) * 14
    bg[:, :, 2] = 26 + (y_grid / H) * 28

    # Top-center cyan spotlight glow
    dist_top = np.sqrt(((x_grid - W * 0.5) / 520.0) ** 2 + ((y_grid - 220.0) / 420.0) ** 2)
    glow1 = np.exp(-dist_top ** 2) * 38.0
    bg[:, :, 1] += glow1 * 0.75
    bg[:, :, 2] += glow1 * 1.15

    # Bottom-center subtle blue glow
    dist_bot = np.sqrt(((x_grid - W * 0.5) / 560.0) ** 2 + ((y_grid - 1650.0) / 380.0) ** 2)
    glow2 = np.exp(-dist_bot ** 2) * 26.0
    bg[:, :, 0] += glow2 * 0.3
    bg[:, :, 1] += glow2 * 0.6
    bg[:, :, 2] += glow2 * 1.0

    # Sleek glowing phone bezel border around (x=96..984, y=130..1790)
    pil_bg = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(pil_bg)
    draw.rounded_rectangle([88, 122, 992, 1798], radius=46, outline=(0, 210, 255, 190), width=5)
    draw.rounded_rectangle([94, 128, 986, 1792], radius=40, outline=(255, 255, 255, 65), width=2)
    return np.array(pil_bg.convert("RGB"))

STUDIO_BG = build_luxury_backdrop()
PHONE_X, PHONE_Y, PHONE_W, PHONE_H = 98, 132, 884, 1656

def prepare_crisp_phone_clip(raw_clip, start_t, end_t, crop_box, subtle_zoom=True):
    """Lanczos4 upscale + Unsharp Mask sharpening inside a 3D glass phone frame so text is razor-sharp."""
    safe_end = min(end_t, max(start_t + 0.5, raw_clip.duration - 0.05))
    sub = raw_clip.subclip(start_t, safe_end).set_fps(30)
    x1, y1, x2, y2 = crop_box
    dur = max(sub.duration, 0.1)

    def render_crisp_frame(get_frame, t):
        frame = get_frame(t)
        cropped = frame[y1:y2, x1:x2]
        # High-precision Lanczos4 resize to fit phone screen without excessive stretch
        ch, cw = cropped.shape[:2]
        scale_base = max(PHONE_W / max(cw, 1), PHONE_H / max(ch, 1))
        zoom_factor = 1.0 + (0.035 * min(max(t / dur, 0.0), 1.0) if subtle_zoom else 0.0)
        nw = int(cw * scale_base * zoom_factor)
        nh = int(ch * scale_base * zoom_factor)
        upscaled = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

        # Center crop to exact PHONE_W x PHONE_H
        cx0 = max(0, (nw - PHONE_W) // 2)
        cy0 = max(0, (nh - PHONE_H) // 2)
        screen = upscaled[cy0:cy0 + PHONE_H, cx0:cx0 + PHONE_W]
        if screen.shape[0] != PHONE_H or screen.shape[1] != PHONE_W:
            screen = cv2.resize(screen, (PHONE_W, PHONE_H), interpolation=cv2.INTER_LANCZOS4)

        # Unsharp Mask sharpening so small UI text & numbers look crisp and clear
        blurred = cv2.GaussianBlur(screen, (0, 0), sigmaX=1.4)
        sharpened = cv2.addWeighted(screen, 1.55, blurred, -0.55, 0)

        # Composite onto studio dark-glass phone backdrop
        canvas = STUDIO_BG.copy()
        canvas[PHONE_Y:PHONE_Y + PHONE_H, PHONE_X:PHONE_X + PHONE_W] = sharpened
        return canvas

    return sub.fl(render_crisp_frame)

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
    """Renders modern 2-line commercial punch captions with pop-in animation."""
    canvas_w, canvas_h = 920, 240
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
        font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
    except Exception:
        font1 = font2 = ImageFont.load_default()

    b1 = draw.textbbox((0, 0), line1, font=font1)
    b2 = draw.textbbox((0, 0), line2, font=font2)
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]

    box_w = min(canvas_w - 16, max(w1, w2) + 76)
    box_h = h1 + h2 + 58
    bx = (canvas_w - box_w) // 2
    by = (canvas_h - box_h) // 2

    draw.rounded_rectangle(
        [bx, by, bx + box_w, by + box_h],
        radius=28,
        fill=(6, 12, 26, 238),
        outline=(0, 215, 255, 255),
        width=4
    )

    x1 = (canvas_w - w1) // 2
    y1 = by + 20
    draw.text((x1 + 2, y1 + 2), line1, font=font1, fill=(0, 0, 0, 255))
    draw.text((x1, y1), line1, font=font1, fill=(235, 245, 255, 255))

    x2 = (canvas_w - w2) // 2
    y2 = y1 + h1 + 14
    c2 = (255, 225, 55, 255) if highlight_second else (0, 225, 255, 255)
    draw.text((x2 + 3, y2 + 3), line2, font=font2, fill=(0, 0, 0, 255))
    draw.text((x2, y2), line2, font=font2, fill=c2)

    base_clip = ImageClip(np.array(img)).set_start(start_t).set_duration(duration)
    # Subtle 0.12s upward slide-in for energetic commercial feel
    return base_clip.set_position(lambda t: (45, int(1345 + max(0.0, (0.12 - t) / 0.12) * 28)))

def prepare_studio_voice(voice_path, max_spoken_seconds=28.1):
    """Trims start/end silence, masters to -14 LUFS, and guarantees final word ('portfolio') never cuts off."""
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
        print(f"⚡ Applying gentle {tempo:.3f}x tempo fit so final syllable finishes cleanly...")
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
    return AudioFileClip(bed_wav).volumex(0.35)

def build_masterpiece():
    print("🚀 Starting Commercial-Grade 100/100 FaucetPay Short Render...")
    screen_path = find_asset("screen_record*", required=True)
    voice_path = find_asset("voice*", required=True)

    crop_box = detect_mobile_viewport(screen_path)
    print(f"📐 Detected Mobile Viewport Crop Box: {crop_box}")

    raw_screen = VideoFileClip(screen_path).without_audio()

    # 1. Exact Voice-Synced 30.0s Commercial Timeline
    # [0.0 - 1.5s] Official FaucetPay 3D Brand Intro
    intro_path = find_asset("fp-stream-intro-4s*", required=False)
    if intro_path:
        intro_raw = VideoFileClip(intro_path).without_audio()
        c_intro = intro_raw.subclip(0, min(1.5, intro_raw.duration)).resize((W, H)).set_fps(30)
    else:
        c_intro = prepare_crisp_phone_clip(raw_screen, 0.0, 1.5, crop_box, subtle_zoom=False)

    # [1.5 - 4.5s] Hook: Real $8.62 Live Dashboard ("Stop letting massive gas fees...")
    c_hook_real = prepare_crisp_phone_clip(raw_screen, 0.0, 3.0, crop_box, subtle_zoom=True)

    # [4.5 - 10.5s] Step 1: Official 3D Claim B-roll (3.0s) + Real Verified Faucets & PTC Ads (3.0s)
    # Matches voice: "First, collect free crypto drops from verified faucets and quick ads..."
    c_broll_claim = prepare_native_broll(
        find_asset("fp-broll-ui-claim*", required=False), 3.0,
        prepare_crisp_phone_clip(raw_screen, 67.0, 70.0, crop_box, subtle_zoom=True)
    )
    c_real_ptc = prepare_crisp_phone_clip(raw_screen, 70.0, 73.0, crop_box, subtle_zoom=True)

    # [10.5 - 16.5s] Step 2: Official 3D Balance B-roll (2.5s) + Real 20+ Coins Wallet (3.5s)
    # Matches voice: "Next, hold over 20 different coins in one dashboard..."
    c_broll_bal = prepare_native_broll(
        find_asset("fp-broll-ui-balance*", required=False), 2.5,
        prepare_crisp_phone_clip(raw_screen, 13.0, 15.5, crop_box, subtle_zoom=True)
    )
    c_real_wallet = prepare_crisp_phone_clip(raw_screen, 14.5, 18.0, crop_box, subtle_zoom=True)

    # [16.5 - 23.5s] Step 3: Real Live Instant Swap USDT->LTC (5.0s) + Official 3D Swap B-roll (2.0s)
    # Matches voice: "Then instantly swap assets like USDT to Litecoin with zero external fees."
    c_real_swap = prepare_crisp_phone_clip(raw_screen, 53.5, 58.5, crop_box, subtle_zoom=True)
    c_broll_swap = prepare_native_broll(
        find_asset("fp-broll-ui-swap*", required=False), 2.0,
        prepare_crisp_phone_clip(raw_screen, 58.5, 60.5, crop_box, subtle_zoom=False)
    )

    # [23.5 - 26.0s] Real Dashboard Trust Close + [26.0 - 30.0s] Official FaucetPay 9:16 Outro
    # Matches voice: "Skipping gas fees makes collecting small daily payouts the safest way..."
    c_close_real = prepare_crisp_phone_clip(raw_screen, 0.5, 3.0, crop_box, subtle_zoom=False)
    outro_raw = VideoFileClip(find_asset("fp-stream-outro-5s*", required=True)).without_audio()
    c_outro = outro_raw.subclip(0, min(4.0, outro_raw.duration)).resize((W, H)).set_fps(30)

    base_timeline = concatenate_videoclips(
        [c_intro, c_hook_real, c_broll_claim, c_real_ptc, c_broll_bal, c_real_wallet, c_real_swap, c_broll_swap, c_close_real, c_outro],
        method="compose"
    ).set_duration(30.0)

    layers = [base_timeline]

    # 2. Official Brand Badge & Animated Stickers (Synced to exact features)
    logo_pop_path = find_asset("fp-edit-sticker-logo-pop*", required=False)
    if logo_pop_path:
        layers.append(
            smart_key_sticker(logo_pop_path, 400)
            .set_start(1.6)
            .set_duration(2.42)
            .set_position(("center", 230))
        )

    logo_dark_path = find_asset("fp-logo-full-on-dark*", required=False)
    if logo_dark_path:
        layers.append(
            ImageClip(logo_dark_path)
            .resize(width=230)
            .set_start(4.1)
            .set_duration(21.9)
            .set_position((W - 285, 58))
        )

    # Spinning FaucetPay mark during Faucets & Ads (7.5s)
    mark_spin_path = find_asset("fp-edit-sticker-mark-spin*", required=False)
    if mark_spin_path:
        layers.append(
            smart_key_sticker(mark_spin_path, 210)
            .set_start(7.5)
            .set_duration(2.42)
            .set_position((65, 240))
        )

    # Claim/Swap Pop right when Live USDT -> LTC Swap executes (18.5s)
    claim_pop_path = find_asset("fp-edit-sticker-claim-pop*", required=False)
    if claim_pop_path:
        layers.append(
            smart_key_sticker(claim_pop_path, 430)
            .set_start(18.5)
            .set_duration(2.42)
            .set_position(("center", 250))
        )

    # 3. Voice-Synced Commercial 2-Line Punch Captions
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

    # 5. Broadcast Studio Audio Mix (Zero-Cutoff Voice + Ambient Bed + Official SFX)
    audio_tracks = []
    voice = prepare_studio_voice(voice_path, max_spoken_seconds=28.1)
    audio_tracks.append(voice.set_start(0.25).volumex(1.15))
    audio_tracks.append(build_ambient_bed(30.0))

    sonic_path = find_asset("fp-audio-sonic-logo*", required=False)
    if sonic_path:
        audio_tracks.append(AudioFileClip(sonic_path).set_start(0.0).volumex(0.50))

    drip_path = find_asset("fp-audio-sfx-coin-drip*", required=False, exclude_word="cascade")
    if drip_path:
        audio_tracks.append(AudioFileClip(drip_path).set_start(7.5).volumex(0.60))

    cascade_path = find_asset("fp-audio-sfx-coin-drip-cascade*", required=False)
    if cascade_path:
        audio_tracks.append(AudioFileClip(cascade_path).set_start(18.5).volumex(0.68))

    final_audio = CompositeAudioClip(audio_tracks).set_duration(30.0)
    final_video = final_video.set_audio(final_audio)

    output_name = "final_faucetpay_shorts.mp4"
    print(f"🎬 Exporting {output_name} (1080x1920, 30fps, 10Mbps Commercial Master)...")
    final_video.write_videofile(
        output_name,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="10000k",
        threads=4,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )
    print("✨ Commercial-Grade 100/100 Short Rendered Successfully!")

if __name__ == "__main__":
    build_masterpiece()
