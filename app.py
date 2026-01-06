import streamlit as st
from PIL import Image, ImageOps
import mediapipe as mp
import numpy as np
import io
import zipfile
import os
from pdf2image import convert_from_bytes

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="Profile Photo Cropper")

# --- 定数 ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- 関数定義 ---

def load_image(uploaded_file):
    """ファイルを読み込みPIL Imageに変換"""
    try:
        if uploaded_file.type == "application/pdf":
            # 修正: DPIを200 -> 300に変更して画質向上（顔認識精度の改善）
            # 300dpiあればA4サイズでも顔の輪郭がくっきりし、過剰なズームを防げます
            images = convert_from_bytes(uploaded_file.getvalue(), dpi=300, fmt='jpeg')
            return images[0] if images else None
        else:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image) # 回転補正
            return image
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return None

def analyze_face_coordinates(image):
    """画像から顔と目の座標(相対値 0.0-1.0)を抽出"""
    img_np = np.array(image.convert('RGB'))
    results = face_detection.process(img_np)
    
    if not results.detections:
        return None

    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    kps = detection.location_data.relative_keypoints
    
    # 目の中心位置
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
    """
    要件に基づきリサイズ・切り抜き・余白処理を行った画像を生成して返す
    優先順位:
    1. 指定の顔サイズ・目線位置で計算
    2. 画像からはみ出る場合、位置調整(Shift)と拡大(Zoom 15%以内)ではみ出しを解消トライ
    3. それでも無理なら指定背景(白/黒)でパディング
    """
    img_w, img_h = original_img.size
    
    # --- 1. 理想の切り抜き枠を計算 ---
    if face_data:
        # 顔の高さ(px) / 設定比率 = 切り抜き枠の高さ
        face_h_px = face_data['face_h'] * img_h
        crop_h = face_h_px / face_ratio
        
        # 目の位置合わせ
        eye_y_px = face_data['eye_cy'] * img_h
        # 枠の上辺(top) = 目のY座標 - (枠の高さ * 目線の設定%)
        crop_top = eye_y_px - (crop_h * eye_level)
        crop_cy = crop_top + (crop_h / 2)
        
        # 横位置は顔中心
        crop_cx = face_data['face_cx'] * img_w
    else:
        # 顔がない場合は画像中心
        crop_h = img_h * 0.8
        crop_cx, crop_cy = img_w / 2, img_h / 2

    # アスペクト比から幅を計算
    target_aspect = target_w / target_h
    crop_w = crop_h * target_aspect
    
    # 理想の座標 (x1, y1, x2, y2)
    x1 = crop_cx - crop_w / 2
    y1 = crop_cy - crop_h / 2
    x2 = crop_cx + crop_w / 2
    y2 = crop_cy + crop_h / 2
    
    # --- 2. 15%ルールによる自動調整 (Shift & Zoom) ---
    
    # 現在の枠が画像内に収まっているかチェック
    # はみ出し量(正の値ならはみ出している)
    overflow_left = max(0, -x1)
    overflow_right = max(0, x2 - img_w)
    overflow_top = max(0, -y1)
    overflow_bottom = max(0, y2 - img_h)
    
    has_overflow = (overflow_left + overflow_right + overflow_top + overflow_bottom) > 0
    
    # 最終的な切り抜きパラメータ
    final_x1, final_y1, final_x2, final_y2 = x1, y1, x2, y2
    needs_padding = False
    
    if has_overflow:
        # A. まず位置ずらし(Shift)で解決できるか？
        # 幅が画像より小さいなら、左右に動かして収める
        if crop_w <= img_w:
            if final_x1 < 0: # 左にはみ出してる -> 右にずらす
                offset = -final_x1
                final_x1 += offset
                final_x2 += offset
            elif final_x2 > img_w: # 右にはみ出してる -> 左にずらす
                offset = final_x2 - img_w
                final_x1 -= offset
                final_x2 -= offset
        
        # 高さが画像より小さいなら、上下に動かして収める
        if crop_h <= img_h:
            if final_y1 < 0:
                offset = -final_y1
                final_y1 += offset
                final_y2 += offset
            elif final_y2 > img_h:
                offset = final_y2 - img_h
                final_y1 -= offset
                final_y2 -= offset

        # ずらした後、まだはみ出しているか再チェック (画像サイズ自体が足りない場合など)
        overflow_w = max(0, -final_x1) + max(0, final_x2 - img_w)
        overflow_h = max(0, -final_y1) + max(0, final_y2 - img_h)
        
        if overflow_w > 0 or overflow_h > 0:
            # B. ズーム(枠を縮小)して解決できるか？ (15%ルール)
            # 現在の枠サイズに対して、画像内に収めるために必要な縮小率を計算
            # width_scale: 画像幅 / 現在の枠幅 (これが1.0以下なら縮小必要)
            scale_w = img_w / crop_w if crop_w > img_w else 1.0
            scale_h = img_h / crop_h if crop_h > img_h else 1.0
            
            min_scale = min(scale_w, scale_h)
            
            # 許容範囲: 「15%の拡大」= 枠を 1 / 1.15 倍 (約0.87) まで小さくしていい
            ALLOWED_SHRINK_LIMIT = 1.0 / 1.15
            
            if min_scale >= ALLOWED_SHRINK_LIMIT:
                # 許容範囲内なので、枠を縮小(Zoom In)してフィットさせる
                new_crop_w = crop_w * min_scale
                new_crop_h = crop_h * min_scale
                
                # 中心を維持しつつリサイズ
                center_x = (final_x1 + final_x2) / 2
                center_y = (final_y1 + final_y2) / 2
                
                # 画像外にはみ出さないよう中心も再クランプ
                center_x = max(new_crop_w/2, min(img_w - new_crop_w/2, center_x))
                center_y = max(new_crop_h/2, min(img_h - new_crop_h/2, center_y))
                
                final_x1 = center_x - new_crop_w / 2
                final_y1 = center_y - new_crop_h / 2
                final_x2 = center_x + new_crop_w / 2
                final_y2 = center_y + new_crop_h / 2
                
                # これでPadding不要
                needs_padding = False
            else:
                # 15%を超えてしまう -> あきらめてパディング(背景追加)する
                # ただし、できるだけ画像を入れるために、枠は理想サイズのままにする
                # (上で計算したIdeal Boxを使う)
                final_x1, final_y1, final_x2, final_y2 = x1, y1, x2, y2
                needs_padding = True
    
    # --- 3. 画像生成実行 ---
    
    if not needs_padding:
        # 通常の切り抜き & リサイズ
        box = (final_x1, final_y1, final_x2, final_y2)
        # 座標を画像内に収める(念のため)
        cx1 = max(0, box[0])
        cy1 = max(0, box[1])
        cx2 = min(img_w, box[2])
        cy2 = min(img_h, box[3])
        
        cropped = original_img.crop((cx1, cy1, cx2, cy2))
        return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    else:
        # パディング処理
        # 背景色決定
        bg_color = (255, 255, 255) if bg_mode == "白" else (0, 0, 0)
        
        # 1. 元画像をアスペクト比維持でターゲット枠内に収まる最大サイズにリサイズ
        #    ターゲットサイズと元画像の比率を比較
        src_aspect = img_w / img_h
        
        if src_aspect > target_aspect:
            # 元画像の方が横長 -> 幅を合わせる
            resize_w = target_w
            resize_h = int(target_w / src_aspect)
        else:
            # 元画像の方が縦長 -> 高さを合わせる
            resize_h = target_h
            resize_w = int(target_h * src_aspect)
            
        resized_src = original_img.resize((resize_w, resize_h), Image.Resampling.LANCZOS)
        
        # 2. ベース作成
        new_img = Image.new("RGB", (target_w, target_h), bg_color)
        
        # 3. 中央配置
        paste_x = (target_w - resize_w) // 2
        paste_y = (target_h - resize_h) // 2
        
        new_img.paste(resized_src, (paste_x, paste_y))
        return new_img

# --- セッション初期化 ---
if 'images_data' not in st.session_state:
    st.session_state['images_data'] = {} 

# --- サイドバー設定 ---
st.sidebar.title("設定")

# 1. 出力サイズ
st.sidebar.subheader("① 出力サイズ")
col_w, col_h = st.sidebar.columns(2)
target_w = col_w.number_input("幅 (px)", value=600, step=10)
target_h = col_h.number_input("高さ (px)", value=800, step=10)

st.sidebar.markdown("---")

# 2. 構図調整
st.sidebar.subheader("② 構図調整")
face_ratio = st.sidebar.slider(
    "顔の大きさ (Zoom)", 0.2, 0.8, 0.45, 0.01,
    help="値が大きいほど顔がアップになります。"
)
eye_level = st.sidebar.slider(
    "目の高さ (上下位置)", 0.2, 0.6, 0.40, 0.01,
    help="上からの位置(%)。値が小さいほど頭上の余白が広くなります。"
)

# 3. 余白処理
st.sidebar.subheader("③ 余白処理")
st.sidebar.caption("拡大・移動(15%以内)で調整できない場合の背景色")
bg_mode = st.sidebar.radio("背景色", ["白", "黒"], horizontal=True)

st.sidebar.markdown("---")

# 4. ダウンロード
st.sidebar.subheader("④ 出力")
if st.session_state['images_data']:
    if st.sidebar.button("📦 画像を作成してダウンロード", type="primary"):
        zip_buffer = io.BytesIO()
        progress_bar = st.sidebar.progress(0)
        total = len(st.session_state['images_data'])
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, (fname, data) in enumerate(st.session_state['images_data'].items()):
                # 高画質生成
                final_img = create_smart_cropped_image(
                    data['original'], data['face_data'],
                    target_w, target_h, face_ratio, eye_level, bg_mode
                )
                
                img_byte_arr = io.BytesIO()
                final_img.save(img_byte_arr, format='JPEG', quality=95)
                zf.writestr(f"{fname}.jpg", img_byte_arr.getvalue())
                
                progress_bar.progress((i + 1) / total)
        
        st.sidebar.success("完了！")
        st.sidebar.download_button(
            label="ZIPファイルを保存",
            data=zip_buffer.getvalue(),
            file_name="processed_photos.zip",
            mime="application/zip"
        )

# --- メイン画面 ---
st.title("プロフィール写真 自動クロッパー")
st.info("設定を変えると、自動的に「位置調整・拡大・余白追加」を判断してプレビューを更新します。")

uploaded_files = st.file_uploader(
    "画像をドラッグ＆ドロップ", type=['jpg', 'jpeg', 'png', 'pdf'], accept_multiple_files=True
)

if uploaded_files:
    new_count = 0
    for up_file in uploaded_files:
        fname = os.path.splitext(up_file.name)[0]
        if fname not in st.session_state['images_data']:
            img = load_image(up_file)
            if img:
                if img.mode != "RGB": img = img.convert("RGB")
                face_data = analyze_face_coordinates(img)
                st.session_state['images_data'][fname] = {'original': img, 'face_data': face_data}
                new_count += 1
    if new_count > 0:
        st.success(f"{new_count} 枚追加しました")

# --- プレビュー ---
if st.session_state['images_data']:
    st.subheader("プレビュー (出力イメージ)")
    cols = st.columns(4)
    keys = list(st.session_state['images_data'].keys())
    
    for i, key in enumerate(keys):
        data = st.session_state['images_data'][key]
        
        # プレビュー生成 (リサイズ済みの画像が返ってくる)
        # 高速化のため、元画像自体を少し縮小してから渡してもよいが、
        # Streamlit Cloudの性能ならこのままでも数枚程度なら許容範囲
        preview_img = create_smart_cropped_image(
            data['original'], data['face_data'],
            target_w, target_h, face_ratio, eye_level, bg_mode
        )
        
        with cols[i % 4]:
            # 画像の形は指定サイズ(のアスペクト比)に統一されて表示される
            st.image(preview_img, caption=key, use_column_width=True)

