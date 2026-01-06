import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import mediapipe as mp
import numpy as np
import io
import zipfile
import os
from pdf2image import convert_from_bytes

# --- ページ設定 ---
st.set_page_config(layout="wide", page_title="Profile Photo Cropper")

# --- 定数・初期設定 ---
TARGET_W_DEFAULT = 600
TARGET_H_DEFAULT = 800
FACE_RATIO_DEFAULT = 0.45  # 画像の高さに対して顔が占める割合（バストアップ用）

# MediaPipeの設定
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- 関数定義 ---

def load_image(uploaded_file):
    """アップロードされたファイルをPIL Imageとして読み込む（PDF対応）"""
    try:
        if uploaded_file.type == "application/pdf":
            # PDFの場合は1ページ目を画像化 (dpi=200でメモリ節約しつつ品質確保)
            images = convert_from_bytes(uploaded_file.getvalue(), dpi=200, fmt='jpeg')
            if images:
                return images[0]
            else:
                return None
        else:
            image = Image.open(uploaded_file)
            # iPhone写真などの回転情報を補正
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
            return image
    except Exception as e:
        st.error(f"読み込みエラー: {uploaded_file.name} - {e}")
        return None

def detect_face_and_suggest_box(image, face_ratio):
    """
    MediaPipeを使って顔を検出し、バストアップ構図になるような
    切り抜きボックス（Box）の座標を計算して返す
    """
    img_np = np.array(image)
    h, w, _ = img_np.shape
    results = face_detection.process(img_np)

    if not results.detections:
        # 顔が見つからない場合は画像中央を返す
        return (0, 0, w, h)

    # 最初の顔を取得（複数人の場合は一番確信度が高いもの、あるいは配列の最初）
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    
    # 顔の座標 (ピクセル)
    face_w = int(bboxC.width * w)
    face_h = int(bboxC.height * h)
    face_x = int(bboxC.xmin * w)
    face_y = int(bboxC.ymin * h)
    
    # 顔の中心
    face_center_x = face_x + face_w // 2
    face_center_y = face_y + face_h // 2

    # --- バストアップ構図の計算ロジック ---
    # 指定された「顔の比率(face_ratio)」から、必要な「切り抜き枠の高さ」を逆算
    # Crop Height = Face Height / Ratio
    crop_h = int(face_h / face_ratio)
    
    # 出力アスペクト比に合わせて幅を計算 (UI設定値から取得)
    target_aspect = st.session_state['target_w'] / st.session_state['target_h']
    crop_w = int(crop_h * target_aspect)

    # 切り抜き枠の中心位置を決める
    # バストアップなので、顔の中心は「枠の上から35%〜40%」くらいの位置に来ると自然
    crop_center_y = face_center_y + (crop_h * 0.1) # 少し下にずらす（＝顔が上に来る）

    # 座標計算 (枠外にはみ出さない処理を含む)
    x1 = int(face_center_x - crop_w // 2)
    y1 = int(crop_center_y - crop_h // 2)
    
    # 画像範囲内に収める補正（簡易版）
    # ※厳密にやるとアスペクト比が崩れるため、ここでは座標計算用としてそのまま返すか、
    # st_cropper側で制限させる。今回は初期値計算なので、はみ出し許容して計算値を返す。
    
    # st_cropper用のbox辞書 (left, top, width, height)
    # ※負の値になるとエラーになる場合があるため調整
    box = {
        'left': max(0, x1),
        'top': max(0, y1),
        'width': crop_w,
        'height': crop_h
    }
    return box

def process_crop_and_resize(image, box):
    """指定されたBoxで切り抜き、ターゲットサイズにリサイズする"""
    # Box情報からcrop (st_cropperの戻り値等を使用)
    left = box['left']
    top = box['top']
    width = box['width']
    height = box['height']
    
    # 画像範囲外参照を防ぐ
    img_w, img_h = image.size
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, left + width)
    bottom = min(img_h, top + height)
    
    cropped = image.crop((left, top, right, bottom))
    
    # 指定サイズにリサイズ (Lanczosフィルタで高品質に)
    target_size = (st.session_state['target_w'], st.session_state['target_h'])
    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
    return resized

# --- セッション状態の初期化 ---
if 'processed_images' not in st.session_state:
    st.session_state['processed_images'] = {} # {filename: PIL Image}
if 'editing_file' not in st.session_state:
    st.session_state['editing_file'] = None
if 'original_images' not in st.session_state:
    st.session_state['original_images'] = {} # {filename: PIL Image (Original)}

# --- サイドバー設定 ---
st.sidebar.header("出力設定")
st.session_state['target_w'] = st.sidebar.number_input("出力 幅(px)", value=TARGET_W_DEFAULT, step=10)
st.session_state['target_h'] = st.sidebar.number_input("出力 高さ(px)", value=TARGET_H_DEFAULT, step=10)

st.sidebar.markdown("---")
st.sidebar.header("自動検出設定")
face_ratio_val = st.sidebar.slider(
    "顔の大きさ比率 (バストアップ調整)", 
    min_value=0.2, max_value=0.8, value=FACE_RATIO_DEFAULT, step=0.05,
    help="値が小さいほど引きで(体が入る)、大きいほど顔のアップになります。"
)

# --- メインエリア ---
st.title("🏆 プロフィール写真 自動＆手動クロッパー")
st.info("PDFまたは画像をアップロードしてください。AIが自動でバストアップ構図を作成します。その後、手動で微調整が可能です。")

uploaded_files = st.file_uploader(
    "画像ファイルをドラッグ＆ドロップ (JPG, PNG, PDF)", 
    type=['jpg', 'jpeg', 'png', 'pdf'], 
    accept_multiple_files=True
)

# ファイルがアップロードされたら処理開始
if uploaded_files:
    # 新規ファイルがあれば読み込んで初期処理
    for uploaded_file in uploaded_files:
        fname = os.path.splitext(uploaded_file.name)[0] # 拡張子なしファイル名
        
        if fname not in st.session_state['original_images']:
            with st.spinner(f'{uploaded_file.name} を読み込み・AI解析中...'):
                # 1. 画像読み込み
                img = load_image(uploaded_file)
                if img is None: continue
                
                # RGB変換
                if img.mode != "RGB":
                    img = img.convert("RGB")

                st.session_state['original_images'][fname] = img
                
                # 2. AIによる初期クロップ位置の計算
                initial_box = detect_face_and_suggest_box(img, face_ratio_val)
                
                # 3. 初期クロップ実行して保存
                processed = process_crop_and_resize(img, initial_box)
                st.session_state['processed_images'][fname] = processed

    st.success(f"{len(st.session_state['processed_images'])} 枚の画像を処理しました。")
    st.markdown("---")

    # --- 編集モード or 一覧モードの切り替え ---

    if st.session_state['editing_file']:
        # === 個別編集モード (Cropper表示) ===
        target_file = st.session_state['editing_file']
        original_img = st.session_state['original_images'][target_file]
        
        st.subheader(f"編集モード: {target_file}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # ターゲットのアスペクト比
            aspect_ratio = (st.session_state['target_w'], st.session_state['target_h'])
            
            # Cropperの表示
            # realtime_update=Trueだと重いのでFalse推奨だが、使い勝手のためTrueにする場合は注意
            cropped_img = st_cropper(
                original_img,
                realtime_update=True,
                box_color='blue',
                aspect_ratio=aspect_ratio,
                should_resize_image=True # 表示を画面内に収める
            )
            
        with col2:
            st.write("プレビュー (リサイズ後)")
            # 確定前のプレビュー表示（指定サイズにリサイズしてみる）
            preview_resized = cropped_img.resize(
                (st.session_state['target_w'], st.session_state['target_h']), 
                Image.Resampling.LANCZOS
            )
            st.image(preview_resized)
            
            st.markdown("### 操作")
            if st.button("✅ この構図で確定する", type="primary"):
                # 編集結果を保存して一覧に戻る
                st.session_state['processed_images'][target_file] = preview_resized
                st.session_state['editing_file'] = None
                st.rerun()
            
            if st.button("キャンセル"):
                st.session_state['editing_file'] = None
                st.rerun()

    else:
        # === 一覧（ギャラリー）モード ===
        st.subheader("処理結果一覧")
        
        # グリッド表示のための列設定
        cols = st.columns(4) # 4列表示
        keys = list(st.session_state['processed_images'].keys())
        
        for i, key in enumerate(keys):
            img = st.session_state['processed_images'][key]
            with cols[i % 4]:
                st.image(img, caption=key, use_container_width=True)
                if st.button(f"編集 ✏️", key=f"edit_{key}"):
                    st.session_state['editing_file'] = key
                    st.rerun()

        st.markdown("---")
        
        # === 一括ダウンロード ===
        st.header("ダウンロード")
        
        # ZIP作成
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for key, img in st.session_state['processed_images'].items():
                # メモリ上の画像をJPGバイト列に変換
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=95)
                # ZIPに追加 (拡張子は.jpg固定)
                zf.writestr(f"{key}.jpg", img_byte_arr.getvalue())
        
        st.download_button(
            label="📦 すべての画像をZIPでダウンロード",
            data=zip_buffer.getvalue(),
            file_name="profile_photos.zip",
            mime="application/zip"
        )