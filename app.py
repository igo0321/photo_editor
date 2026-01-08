import streamlit as st
from PIL import Image, ImageOps
import mediapipe as mp
import numpy as np
import io
import zipfile
import os
import gc  # メモリ掃除用
from pdf2image import convert_from_bytes

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="Profile Photo Cropper")

# --- 定数 ---
# メモリ対策: 作業用画像の最大サイズ(長辺px)
# 出力サイズが800px程度なら、2000pxあればズームしても十分高画質を維持でき、かつメモリを節約できる
MAX_WORKING_SIZE = 2000 

# --- 関数定義 ---

def resize_if_huge(image):
    """画像が巨大すぎる場合、アスペクト比を維持してリサイズする"""
    w, h = image.size
    max_dim = max(w, h)
    if max_dim > MAX_WORKING_SIZE:
        scale = MAX_WORKING_SIZE / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image

def load_image(uploaded_file):
    """ファイルを読み込みPIL Imageに変換 (メモリ対策込み)"""
    try:
        image = None
        if uploaded_file.type == "application/pdf":
            # PDFは300dpiで変換して顔認識精度を確保
            images = convert_from_bytes(uploaded_file.getvalue(), dpi=300, fmt='jpeg')
            if images:
                image = images[0]
        else:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image) # 回転補正
        
        if image:
            # ここで巨大画像をリサイズしてメモリ爆発を防ぐ
            image = resize_if_huge(image)
            return image
        return None
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return None

def analyze_face_coordinates(image, confidence_threshold):
    """指定された感度(confidence)で顔検出を行う"""
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=confidence_threshold) as face_detection:
        img_np = np.array(image.convert('RGB'))
        results = face_detection.process(img_np)
        
        # メモリ開放
        del img_np
        
        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        kps = detection.location_data.relative_keypoints
        
        right_eye = kps[0]
        left_eye = kps[1]
        eye_center_x = (right_eye.x + left_eye.x) / 2
        eye_center_y = (right_eye.y + left_eye.y) / 2
        
        return {
            'face_h': bbox.height,
            'face_cx': bbox.xmin + bbox.width / 2,
            'face_cy': bbox.ymin + bbox.height / 2,
            'eye_cy': eye_center_y
        }

def create_smart_cropped_image(original_img, face_data, target_w, target_h, face_ratio, eye_level, bg_mode):
    """最終画像を生成する（15%ルール & パディング処理込み）"""
    img_w, img_h = original_img.size
    
    # 1. 理想の切り抜き枠
    if face_data:
        face_h_px = face_data['face_h'] * img_h
        crop_h = face_h_px / face_ratio
        eye_y_px = face_data['eye_cy'] * img_h
        crop_top = eye_y_px - (crop_h * eye_level)
        crop_cy = crop_top + (crop_h / 2)
        crop_cx = face_data['face_cx'] * img_w
    else:
        crop_h = img_h * 0.8
        crop_cx, crop_cy = img_w / 2, img_h / 2

    target_aspect = target_w / target_h
    crop_w = crop_h * target_aspect
    
    x1, y1, x2, y2 = crop_cx - crop_w/2, crop_cy - crop_h/2, crop_cx + crop_w/2, crop_cy + crop_h/2
    
    # 2. 自動調整 (15%ルール)
    overflow_left = max(0, -x1)
    overflow_right = max(0, x2 - img_w)
    overflow_top = max(0, -y1)
    overflow_bottom = max(0, y2 - img_h)
    has_overflow = (overflow_left + overflow_right + overflow_top + overflow_bottom) > 0
    
    final_x1, final_y1, final_x2, final_y2 = x1, y1, x2, y2
    needs_padding = False
    
    if has_overflow:
        if crop_w <= img_w:
            if final_x1 < 0:
                offset = -final_x1
                final_x1 += offset
                final_x2 += offset
            elif final_x2 > img_w:
                offset = final_x2 - img_w
                final_x1 -= offset
                final_x2 -= offset
        if crop_h <= img_h:
            if final_y1 < 0:
                offset = -final_y1
                final_y1 += offset
                final_y2 += offset
            elif final_y2 > img_h:
                offset = final_y2 - img_h
                final_y1 -= offset
                final_y2 -= offset

        scale_w = img_w / crop_w if crop_w > img_w else 1.0
        scale_h = img_h / crop_h if crop_h > img_h else 1.0
        min_scale = min(scale_w, scale_h)
        ALLOWED_SHRINK_LIMIT = 1.0 / 1.15
        
        if min_scale >= ALLOWED_SHRINK_LIMIT:
            new_crop_w = crop_w * min_scale
            new_crop_h = crop_h * min_scale
            center_x = (final_x1 + final_x2) / 2
            center_y = (final_y1 + final_y2) / 2
            center_x = max(new_crop_w/2, min(img_w - new_crop_w/2, center_x))
            center_y = max(new_crop_h/2, min(img_h - new_crop_h/2, center_y))
            final_x1, final_y1, final_x2, final_y2 = center_x - new_crop_w/2, center_y - new_crop_h/2, center_x + new_crop_w/2, center_y + new_crop_h/2
        else:
            final_x1, final_y1, final_x2, final_y2 = x1, y1, x2, y2
            needs_padding = True
    
    # 3. 生成
    if not needs_padding:
        cx1, cy1, cx2, cy2 = max(0, final_x1), max(0, final_y1), min(img_w, final_x2), min(img_h, final_y2)
        cropped = original_img.crop((cx1, cy1, cx2, cy2))
        return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:
        bg_color = (255, 255, 255) if bg_mode == "白" else (0, 0, 0)
        src_aspect = img_w / img_h
        if src_aspect > target_aspect:
            resize_w, resize_h = target_w, int(target_w / src_aspect)
        else:
            resize_w, resize_h = int(target_h * src_aspect), target_h
        resized_src = original_img.resize((resize_w, resize_h), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (target_w, target_h), bg_color)
        new_img.paste(resized_src, ((target_w - resize_w)//2, (target_h - resize_h)//2))
        return new_img

# --- セッション初期化 ---
if 'images_data' not in st.session_state:
    st.session_state['images_data'] = {} 
if 'last_detection_confidence' not in st.session_state:
    st.session_state['last_detection_confidence'] = 0.5

# --- サイドバー構成 ---
st.sidebar.title("設定")

# 0. メモリ開放ボタン
if st.sidebar.button("🗑️ データをリセット", help="アップロードした画像を全てクリアします"):
    st.session_state['images_data'] = {}
    gc.collect() # 強制メモリ掃除
    st.rerun()

st.sidebar.markdown("---")

# 1. 顔認識設定
st.sidebar.subheader("① 顔認識の精度")
confidence_val = st.sidebar.slider(
    "検出感度 (低いほど検出しやすい)", 
    0.1, 0.9, 0.5, 0.05,
    help="顔が認識されない場合は値を下げてみてください。"
)

# 感度が変更された場合
if abs(confidence_val - st.session_state['last_detection_confidence']) > 0.001:
    if st.session_state['images_data']:
        with st.spinner("新しい感度で顔を再検出中..."):
            for key in st.session_state['images_data']:
                img = st.session_state['images_data'][key]['original']
                new_face_data = analyze_face_coordinates(img, confidence_val)
                st.session_state['images_data'][key]['face_data'] = new_face_data
            gc.collect() # 処理後に掃除
    st.session_state['last_detection_confidence'] = confidence_val
    st.rerun()

st.sidebar.markdown("---")

# 2. 出力サイズ
st.sidebar.subheader("② 出力サイズ")
col_w, col_h = st.sidebar.columns(2)
target_w = col_w.number_input("幅 (px)", value=600, step=10)
target_h = col_h.number_input("高さ (px)", value=800, step=10)

# 3. 構図調整
st.sidebar.subheader("③ 構図調整")
face_ratio = st.sidebar.slider("顔の大きさ (Zoom)", 0.2, 0.8, 0.45, 0.01)
eye_level = st.sidebar.slider("目の高さ (上下位置)", 0.2, 0.6, 0.40, 0.01)

# 4. 余白処理
st.sidebar.subheader("④ 余白処理")
bg_mode = st.sidebar.radio("背景色", ["白", "黒"], horizontal=True)

st.sidebar.markdown("---")

# 5. ダウンロードボタン配置用プレースホルダー
download_placeholder = st.sidebar.empty()


# --- メイン画面 ---
st.title("プロフィール写真 自動クロッパー")

uploaded_files = st.file_uploader(
    "画像をドラッグ＆ドロップ", type=['jpg', 'jpeg', 'png', 'pdf'], accept_multiple_files=True
)

if uploaded_files:
    new_count = 0
    # プログレスバーを表示（大量アップロード時のフリーズ防止感）
    progress_text = "画像を読み込み中..."
    my_bar = st.progress(0, text=progress_text)
    
    total_files = len(uploaded_files)
    
    for i, up_file in enumerate(uploaded_files):
        fname = os.path.splitext(up_file.name)[0]
        if fname not in st.session_state['images_data']:
            img = load_image(up_file)
            if img:
                if img.mode != "RGB": img = img.convert("RGB")
                face_data = analyze_face_coordinates(img, confidence_val)
                st.session_state['images_data'][fname] = {'original': img, 'face_data': face_data}
                new_count += 1
        
        # 進捗更新
        my_bar.progress((i + 1) / total_files, text=f"読み込み中... {i+1}/{total_files}")
    
    my_bar.empty()
    gc.collect() # 読み込み完了後に一回掃除
    
    if new_count > 0:
        st.success(f"{new_count} 枚追加しました")

# --- プレビュー表示 ---
if st.session_state['images_data']:
    st.subheader("プレビュー (出力イメージ)")
    cols = st.columns(4)
    keys = list(st.session_state['images_data'].keys())
    
    for i, key in enumerate(keys):
        data = st.session_state['images_data'][key]
        preview_img = create_smart_cropped_image(
            data['original'], data['face_data'],
            target_w, target_h, face_ratio, eye_level, bg_mode
        )
        with cols[i % 4]:
            st.image(preview_img, caption=key, use_column_width=True)

    # --- ダウンロードボタン ---
    with download_placeholder.container():
        st.subheader("⑤ 出力")
        if st.button("📦 画像を作成してダウンロード", type="primary"):
            zip_buffer = io.BytesIO()
            progress_bar = st.progress(0)
            status_text = st.empty()
            total = len(st.session_state['images_data'])
            
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for i, (fname, data) in enumerate(st.session_state['images_data'].items()):
                    status_text.text(f"処理中: {fname}...")
                    final_img = create_smart_cropped_image(
                        data['original'], data['face_data'],
                        target_w, target_h, face_ratio, eye_level, bg_mode
                    )
                    img_byte_arr = io.BytesIO()
                    final_img.save(img_byte_arr, format='JPEG', quality=95)
                    zf.writestr(f"{fname}.jpg", img_byte_arr.getvalue())
                    progress_bar.progress((i + 1) / total)
                    
                    # 1枚ごとにメモリ掃除
                    del final_img
                    del img_byte_arr
                    if i % 5 == 0: gc.collect()
            
            progress_bar.empty()
            status_text.empty()
            gc.collect()
            
            st.success("作成完了！下のボタンから保存してください")
            st.download_button(
                label="ZIPファイルを保存",
                data=zip_buffer.getvalue(),
                file_name="processed_photos.zip",
                mime="application/zip"
            )
else:
    download_placeholder.info("画像をアップロードするとダウンロードボタンが表示されます")
