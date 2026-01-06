import streamlit as st
from PIL import Image
import mediapipe as mp
import numpy as np
import io
import zipfile
import os
from pdf2image import convert_from_bytes

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="Profile Photo Cropper")

# --- 定数 ---
# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- 関数定義 ---

def load_image(uploaded_file):
    """ファイルを読み込みPIL Imageに変換"""
    try:
        if uploaded_file.type == "application/pdf":
            # PDFは1ページ目を画像化 (dpi=200)
            images = convert_from_bytes(uploaded_file.getvalue(), dpi=200, fmt='jpeg')
            return images[0] if images else None
        else:
            image = Image.open(uploaded_file)
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
            return image
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return None

def analyze_face_coordinates(image):
    """
    画像から顔と目の座標(相対値 0.0-1.0)を抽出して保存する。
    描画処理は行わない（軽量化のため）。
    """
    img_np = np.array(image.convert('RGB'))
    results = face_detection.process(img_np)
    
    if not results.detections:
        return None

    # 最も確信度の高い顔を1つ取得
    detection = results.detections[0]
    
    # バウンディングボックス (顔の範囲)
    bbox = detection.location_data.relative_bounding_box
    # キーポイント (0:右目, 1:左目, 2:鼻, 3:口, 4:右耳, 5:左耳)
    kps = detection.location_data.relative_keypoints
    
    # 目の中心位置を計算
    right_eye = kps[0]
    left_eye = kps[1]
    eye_center_x = (right_eye.x + left_eye.x) / 2
    eye_center_y = (right_eye.y + left_eye.y) / 2
    
    return {
        'face_h': bbox.height,    # 顔の高さ(比率)
        'face_cx': bbox.xmin + bbox.width / 2, # 顔の中心X
        'face_cy': bbox.ymin + bbox.height / 2, # 顔の中心Y
        'eye_cy': eye_center_y    # 目の高さY(比率)
    }

def get_crop_box(img_w, img_h, face_data, target_w, target_h, face_ratio_setting, eye_level_setting):
    """
    設定値に基づいて切り抜き範囲(絶対座標)を計算する
    """
    if face_data is None:
        # 顔がない場合は画像の中心を切り抜く
        cx, cy = img_w / 2, img_h / 2
        # デフォルトで画像の高さの80%を使う等の処理
        crop_h = img_h * 0.8
    else:
        # 1. 基準となる「切り抜き後の高さ(crop_h)」を決定
        #    設定: 「切り抜き後の画像の中で、顔の高さが face_ratio_setting (例:0.4) を占める」
        #    計算: face_height_px / crop_h = face_ratio_setting
        face_h_px = face_data['face_h'] * img_h
        crop_h = face_h_px / face_ratio_setting
        
        # 2. 目の位置合わせ
        #    設定: 「切り抜き後の画像の上から eye_level_setting (例:0.4=40%) の位置に目が来る」
        #    計算: eye_y_px - crop_top = crop_h * eye_level_setting
        eye_y_px = face_data['eye_cy'] * img_h
        crop_top = eye_y_px - (crop_h * eye_level_setting)
        crop_cy = crop_top + (crop_h / 2)
        
        # 横方向は顔の中心(face_cx)に合わせる
        crop_cx = face_data['face_cx'] * img_w
        cx, cy = crop_cx, crop_cy

    # 3. アスペクト比から幅を計算
    target_aspect = target_w / target_h
    crop_w = crop_h * target_aspect

    # 4. 座標計算 (画像の範囲外処理はcrop時に行うが、ここでは計算上の枠を出す)
    x1 = cx - (crop_w / 2)
    y1 = cy - (crop_h / 2)
    x2 = cx + (crop_w / 2)
    y2 = cy + (crop_h / 2)

    return (x1, y1, x2, y2)

def process_final_image(image, box, target_w, target_h):
    """計算されたBoxで実際に切り抜き＆リサイズを行う"""
    img_w, img_h = image.size
    x1, y1, x2, y2 = box
    
    # 余白が必要な場合（画像外にはみ出している場合）の処理
    # PILのcropは範囲外を指定しても自動で埋めてくれないため、
    # 一度キャンバスを広げるか、padする処理が必要。
    # 簡易的に、背景をボカして埋める等の処理は重いため、
    # ここでは「黒/白で埋める」または「はみ出しを許容して引き伸ばす」ではなく
    # 「範囲内に収まるように移動・縮小」せず、「余白を拡張して切り抜く」アプローチをとります。
    
    # 整数座標へ
    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
    
    # 切り出し用の空画像を作成（背景白）
    crop_w_int = ix2 - ix1
    crop_h_int = iy2 - iy1
    
    # 元画像から切り出せる範囲を計算
    src_x1 = max(0, ix1)
    src_y1 = max(0, iy1)
    src_x2 = min(img_w, ix2)
    src_y2 = min(img_h, iy2)
    
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        # 万が一範囲がおかしい場合はリサイズのみで返す
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    src_crop = image.crop((src_x1, src_y1, src_x2, src_y2))
    
    # 貼り付け位置
    dst_x = src_x1 - ix1
    dst_y = src_y1 - iy1
    
    # ベース作成（白背景）
    # ※将来的にはボカし背景などが選択可能だが今回は白固定
    base = Image.new('RGB', (crop_w_int, crop_h_int), (255, 255, 255))
    base.paste(src_crop, (dst_x, dst_y))
    
    # 最終リサイズ
    final_img = base.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return final_img

# --- セッション初期化 ---
if 'images_data' not in st.session_state:
    st.session_state['images_data'] = {} 
    # { filename: {'original': PILImage, 'face_data': dict} }

# --- サイドバー (設定 & DL) ---
st.sidebar.title("設定")

# 1. 出力サイズ設定
st.sidebar.subheader("① 出力サイズ")
col_w, col_h = st.sidebar.columns(2)
target_w = col_w.number_input("幅 (px)", value=600, step=10)
target_h = col_h.number_input("高さ (px)", value=800, step=10)

st.sidebar.markdown("---")

# 2. 構図調整 (リアルタイム)
st.sidebar.subheader("② 構図調整")
face_ratio = st.sidebar.slider(
    "顔の大きさ (Zoom)", 
    0.2, 0.8, 0.45, 0.01,
    help="画像全体に対する顔の高さの割合。値が大きいほど顔がアップになります。"
)
eye_level = st.sidebar.slider(
    "目の高さ (上下位置)", 
    0.2, 0.6, 0.40, 0.01,
    help="画像の上辺から何%の位置に目を配置するか。値が小さいほど頭上の余白が広くなります。"
)

st.sidebar.markdown("---")

# 3. ダウンロードボタン (サイドバー下部)
st.sidebar.subheader("③ 出力")
if st.session_state['images_data']:
    if st.sidebar.button("📦 画像を作成してダウンロード", type="primary"):
        # ダウンロード処理
        zip_buffer = io.BytesIO()
        progress_bar = st.sidebar.progress(0)
        total = len(st.session_state['images_data'])
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, (fname, data) in enumerate(st.session_state['images_data'].items()):
                # ここで初めて高画質リサイズ処理を行う
                box = get_crop_box(
                    data['original'].width, data['original'].height, data['face_data'],
                    target_w, target_h, face_ratio, eye_level
                )
                final_img = process_final_image(data['original'], box, target_w, target_h)
                
                # 保存
                img_byte_arr = io.BytesIO()
                final_img.save(img_byte_arr, format='JPEG', quality=95)
                zf.writestr(f"{fname}.jpg", img_byte_arr.getvalue())
                
                progress_bar.progress((i + 1) / total)
        
        st.sidebar.success("作成完了！")
        st.sidebar.download_button(
            label="ZIPファイルを保存",
            data=zip_buffer.getvalue(),
            file_name="processed_photos.zip",
            mime="application/zip"
        )
else:
    st.sidebar.info("画像をアップロードしてください")


# --- メイン画面 ---
st.title("プロフィール写真 自動クロッパー")
st.markdown("サイドバーでサイズと構図を調整してください。プレビューは設定に合わせてリアルタイムに変化します。")

uploaded_files = st.file_uploader(
    "画像をドラッグ＆ドロップ (複数可)", 
    type=['jpg', 'jpeg', 'png', 'pdf'], 
    accept_multiple_files=True
)

# 新規アップロード処理
if uploaded_files:
    new_files_count = 0
    for up_file in uploaded_files:
        fname = os.path.splitext(up_file.name)[0]
        if fname not in st.session_state['images_data']:
            img = load_image(up_file)
            if img:
                # RGB変換
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # 顔座標解析 (1回だけ実行)
                face_data = analyze_face_coordinates(img)
                
                st.session_state['images_data'][fname] = {
                    'original': img,
                    'face_data': face_data
                }
                new_files_count += 1
    
    if new_files_count > 0:
        st.success(f"{new_files_count} 枚の画像を追加しました。")

# --- プレビュー表示 ---
if st.session_state['images_data']:
    st.subheader("プレビュー (アスペクト比確認用)")
    st.caption("※表示速度優先のため画質は落としています。ダウンロード時には指定サイズ(px)で高画質出力されます。")
    
    cols = st.columns(4)
    keys = list(st.session_state['images_data'].keys())
    
    for i, key in enumerate(keys):
        data = st.session_state['images_data'][key]
        
        # プレビュー用の計算 (軽い)
        box = get_crop_box(
            data['original'].width, data['original'].height, data['face_data'],
            target_w, target_h, face_ratio, eye_level
        )
        
        # プレビュー表示用に切り抜く (リサイズはブラウザ表示に任せる)
        # 座標を整数に
        x1, y1, x2, y2 = map(int, box)
        
        # 簡易切り抜き（はみ出し処理は簡易的にクリップ）
        img_w, img_h = data['original'].size
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(img_w, x2)
        cy2 = min(img_h, y2)
        
        if cx2 > cx1 and cy2 > cy1:
            preview_img = data['original'].crop((cx1, cy1, cx2, cy2))
        else:
            preview_img = data['original'] # エラー時は元画像

        with cols[i % 4]:
            # use_column_width=Trueでカラム幅に合わせて表示（比率は維持される）
            st.image(preview_img, caption=key, use_column_width=True)
