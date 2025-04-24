import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import time
import json
from pathlib import Path

# 定義口紅色號庫
LIPSTICK_COLORS = {
    'MAC': {
        'Ruby Woo': (185, 25, 25),
        'Chili': (180, 50, 35),
        'Velvet Teddy': (174, 108, 108),
        'Diva': (130, 35, 35),
        'Lady Danger': (228, 50, 30),
        'Russian Red': (190, 30, 30),
        'Mehr': (180, 100, 120)
    },
    'YSL': {
        'Rouge Pur': (170, 20, 60),
        'Le Rouge': (190, 25, 45),
        'Rouge Volupté': (200, 40, 80),
        'Tatouage Couture': (185, 35, 55),
        'Rouge Volupté Shine': (210, 50, 90),
        'Rouge Pur Couture': (195, 30, 50)
    },
    'DIOR': {
        '999 Iconic Red': (205, 20, 40),
        'Rouge Trafalgar': (215, 35, 45),
        'Forever Pink': (230, 130, 150),
        'Rosewood': (175, 95, 95),
        'Forever Nude': (200, 140, 130),
        'Rouge Graphist': (190, 40, 60)
    },
    'CHANEL': {
        'Pirate': (195, 25, 35),
        'Rouge Allure': (185, 30, 45),
        'Camelia': (210, 90, 100),
        'Rouge Coco': (180, 35, 50),
        'Rouge Noir': (90, 20, 25),
        'Boy': (170, 110, 110)
    },
    'NARS': {
        'Dragon Girl': (200, 35, 35),
        'Heat Wave': (230, 60, 40),
        'Dolce Vita': (170, 85, 90),
        'Cruella': (180, 30, 40),
        'Red Square': (210, 45, 35),
        'Jungle Red': (195, 40, 45)
    },
    '3CE': {
        'Taupe': (150, 90, 90),
        'Pink Run': (220, 120, 130),
        'Mellow Flower': (200, 100, 110),
        'Brunch Time': (190, 110, 100),
        'Null Set': (180, 95, 85),
        'Simple Stay': (170, 80, 80)
    },
    'Charlotte Tilbury': {
        'Pillow Talk': (190, 120, 120),
        'Walk of Shame': (180, 70, 80),
        'Red Carpet Red': (200, 30, 40),
        'Bond Girl': (160, 80, 85),
        'Very Victoria': (165, 105, 95),
        'Amazing Grace': (210, 110, 120)
    },
    'Armani': {
        'Red 400': (195, 25, 35),
        'Pink 500': (220, 100, 120),
        'Beige 100': (190, 130, 120),
        'Plum 200': (150, 60, 70),
        'Coral 300': (230, 90, 80),
        'Mauve 600': (170, 90, 100)
    }
}

def detect_lips(image, face_mesh):
    """檢測唇部關鍵點，支持多人臉"""
    # 確保輸入圖像是BGR格式
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.dtype == np.uint8 else image
    else:
        image_bgr = image
        
    h, w = image_bgr.shape[:2]
    max_dimension = 640
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

    # 處理圖像
    results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return None

    # 更新唇部關鍵點索引，使用更精確的點集
    outer_lips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    inner_lips = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    
    all_lips_points = []
    for landmarks in results.multi_face_landmarks:
        lips_points = []
        # 先添加外唇輪廓點
        for idx in outer_lips:
            pt = landmarks.landmark[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            lips_points.append([x, y])
        
        # 再添加內唇輪廓點
        for idx in inner_lips:
            pt = landmarks.landmark[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            lips_points.append([x, y])
        
        all_lips_points.append(np.array(lips_points))

    return all_lips_points

def apply_lipstick(image, lips_points, color, intensity=0.5, texture='matte', lip_reshape=None):
    """應用口紅效果
    Args:
        texture: 口紅質地，可選 'matte'(霧面)、'pearl'(珠光)、'velvet'(絲絨)
        lip_reshape: 唇形修容參數，格式為 (scale_x, scale_y)，用於調整唇形大小
    """
    if lips_points is None:
        return image
    # 確保輸入圖像是RGB格式
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = image if image.dtype == np.uint8 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
        
    # 創建唇部遮罩
    mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    
    # 處理唇形修容
    if lip_reshape:
        scale_x, scale_y = lip_reshape
        center = np.mean(lips_points, axis=0)
        lips_points = (lips_points - center) * [scale_x, scale_y] + center
    
    # 分別處理外唇和內唇
    outer_lips = lips_points[:20]
    inner_lips = lips_points[20:]
    
    # 確保點座標為整數型態的numpy陣列
    outer_lips = np.array(outer_lips, dtype=np.int32)
    inner_lips = np.array(inner_lips, dtype=np.int32)
    
    # 繪製外唇和內唇區域
    cv2.fillPoly(mask, [outer_lips], 255)
    cv2.fillPoly(mask, [inner_lips], 255)
    
    # 應用高斯模糊使邊緣更自然
    mask = cv2.GaussianBlur(mask, (5, 5), 2)
    
    # 創建漸變效果
    result = image_rgb.copy()
    mask_area = mask > 0
    
    # 調整顏色混合方式
    color_layer = np.full_like(image_rgb, color)
    # 使用原始顏色作為基底，增加自然度
    base_color = image_rgb[mask_area].astype(float)
    lip_color = color_layer[mask_area].astype(float)
    
    # 根據質地調整混合效果
    if texture == 'matte':
        # 霧面效果：更強的啞光處理
        matte_intensity = intensity * 1.2  # 增強啞光效果
        blended_color = (base_color * (1 - matte_intensity) + lip_color * matte_intensity).astype(np.uint8)
        # 降低光澤度
        blended_color = np.clip(blended_color * 0.95, 0, 255).astype(np.uint8)
    elif texture == 'pearl':
        # 珠光效果：更明顯的光澤和反射
        highlight = np.clip(lip_color + 80, 0, 255)  # 增強高光
        shimmer = np.random.uniform(0.9, 1.1, base_color.shape)  # 添加細緻光澤變化
        pearl_color = (lip_color * 0.6 + highlight * 0.4) * shimmer
        blended_color = (base_color * (1 - intensity) + pearl_color * intensity).astype(np.uint8)
    else:  # velvet
        # 絲絨效果：更柔和的漸變和質感
        velvet_base = cv2.addWeighted(base_color, 1 - intensity, lip_color, intensity, gamma=0.7)
        # 添加細緻的絲絨質感
        texture_noise = np.random.uniform(0.95, 1.05, base_color.shape)
        blended_color = np.clip(velvet_base * texture_noise, 0, 255).astype(np.uint8)
    
    result[mask_area] = blended_color
    
    return result

def get_skin_tone(image):
    """分析膚色，返回色調值"""
    # 轉換為LAB色彩空間以更準確地分析膚色
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_image)
    # 計算平均值作為膚色特徵
    return np.mean(l), np.mean(a), np.mean(b)

def recommend_lipsticks(skin_tone):
    """根據膚色推薦口紅，使用專業的色彩分析系統"""
    l, a, b = skin_tone
    recommendations = []
    scores = []
    
    # 判斷冷暖色調
    is_warm = b > 0  # b為正值表示偏黃（暖），負值表示偏藍（冷）
    
    # 判斷膚色深淺
    if l < 50:
        skin_depth = "深色"
    elif l < 65:
        skin_depth = "中深色"
    elif l < 80:
        skin_depth = "中色"
    else:
        skin_depth = "淺色"
    
    # 計算色相角度（用於確定適合的色彩）
    hue_angle = np.arctan2(b, a) * 180 / np.pi
    
    for brand, colors in LIPSTICK_COLORS.items():
        for name, rgb in colors.items():
            score = 0
            r, g, b = rgb
            
            # 計算口紅的HSV值以分析其特性
            hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
            hue, sat, val = hsv
            
            # 1. 色調匹配度評分
            if is_warm:
                # 暖色調膚色適合的口紅特徵
                if r > g and r > b:  # 偏紅或珊瑚色
                    score += 2
                if r > 180 and g > 60:  # 帶有金黃調的紅色
                    score += 1
            else:
                # 冷色調膚色適合的口紅特徵
                if r > g and b > g:  # 偏紫紅
                    score += 2
                if b > 50 and r > 150:  # 帶有藍調的紅色
                    score += 1
            
            # 2. 深淺度匹配
            val_norm = val / 255.0
            if skin_depth == "深色" and val_norm < 0.7:
                score += 2
            elif skin_depth == "中深色" and 0.6 <= val_norm <= 0.8:
                score += 2
            elif skin_depth == "中色" and 0.7 <= val_norm <= 0.9:
                score += 2
            elif skin_depth == "淺色" and val_norm > 0.8:
                score += 2
            
            # 3. 飽和度評分
            sat_norm = sat / 255.0
            if skin_depth in ["深色", "中深色"]:
                if sat_norm > 0.7:  # 深色膚色適合高飽和度
                    score += 1
            else:
                if sat_norm < 0.8:  # 淺色膚色適合中等飽和度
                    score += 1
            
            # 4. 季節色彩理論加分
            # 春季：明亮溫暖
            if is_warm and val_norm > 0.8 and sat_norm > 0.7:
                score += 1
            # 夏季：柔和冷調
            elif not is_warm and val_norm > 0.7 and sat_norm < 0.8:
                score += 1
            # 秋季：深沉暖調
            elif is_warm and val_norm < 0.8 and sat_norm > 0.6:
                score += 1
            # 冬季：明亮冷調
            elif not is_warm and val_norm > 0.7 and sat_norm > 0.7:
                score += 1
            
            recommendations.append((brand, name))
            scores.append(score)
    
    # 根據評分排序並返回前5個最佳推薦
    sorted_recommendations = [x for _, x in sorted(zip(scores, recommendations), reverse=True)]
    return sorted_recommendations[:5]  # 返回前5個推薦

def load_favorites():
    """讀取收藏的口紅"""
    fav_file = Path('favorites.json')
    if fav_file.exists():
        return json.loads(fav_file.read_text())
    return []

def save_favorites(favorites):
    """保存收藏的口紅"""
    fav_file = Path('favorites.json')
    fav_file.write_text(json.dumps(favorites))

def main():
    st.title('虛擬口紅試妝系統 💄')
    st.markdown("請上傳自拍照或開啟攝像頭進行試妝。")
    
    # 載入收藏
    favorites = load_favorites()

    input_type = st.radio('選擇輸入方式', ['上傳照片', '使用攝像頭'])

    image = None  # 預先宣告

    # 根據使用方式初始化 FaceMesh
    if input_type == '上傳照片':
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5)
        uploaded_file = st.file_uploader('上傳自拍照', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)
            # PIL圖像默認為RGB格式
    else:
        st.info("⚠️ 請確保臉部清晰、面向鏡頭，並有良好光線")
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=5,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        # 初始化攝像頭
        cap = None
        retry_count = 0
        max_retries = 5  # 增加重試次數
        available_cameras = [0, 1]  # 嘗試不同的攝像頭設備
        
        for camera_id in available_cameras:
            while retry_count < max_retries:
                try:
                    cap = cv2.VideoCapture(camera_id)
                    if cap is not None and cap.isOpened():
                        st.success(f'成功開啟攝像頭 {camera_id}')
                        break
                    retry_count += 1
                    st.warning(f'嘗試開啟攝像頭 {camera_id} 第 {retry_count} 次...')
                    time.sleep(1)
                except Exception as e:
                    st.error(f'開啟攝像頭時發生錯誤: {str(e)}')
                    retry_count += 1
            
            if cap is not None and cap.isOpened():
                break
        
        if cap is None or not cap.isOpened():
            st.error('無法開啟攝像頭，請檢查：\n1. 攝像頭是否正確連接\n2. 是否已授予攝像頭權限\n3. 是否有其他程式正在使用攝像頭')
            return

        # 拍攝照片
        st.write("請等待拍攝中...")
        image = None
        capture_attempts = 5  # 增加拍攝嘗試次數
        
        for attempt in range(capture_attempts):
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    break
                time.sleep(0.5)  # 增加等待時間
            except Exception as e:
                st.warning(f'拍攝第 {attempt + 1} 次失敗: {str(e)}')

        # 釋放攝像頭資源
        try:
            cap.release()
        except Exception as e:
            st.warning(f'釋放攝像頭資源時發生錯誤: {str(e)}')

        if image is None:
            st.error('無法獲取攝像頭畫面，請重試')
            return
            
        st.image(image, caption='拍攝畫面')

        # 保持RGB格式

    # 處理圖像
    if image is not None:
        # 分析膚色並推薦口紅
        skin_tone = get_skin_tone(image)
        recommendations = recommend_lipsticks(skin_tone)
        
        st.subheader('💡 根據您的膚色，我們推薦：')
        for brand, color in recommendations:
            st.write(f'- {brand} {color}')
        
        # 選擇品牌和色號
        selected_brand = st.selectbox('選擇品牌', list(LIPSTICK_COLORS.keys()))
        selected_color = st.selectbox('選擇色號', list(LIPSTICK_COLORS[selected_brand].keys()))
        intensity = st.slider('調整口紅顏色強度', 0.0, 0.2, 0.1)
        
        # 收藏功能
        current_selection = f"{selected_brand} - {selected_color}"
        if st.button('❤️ 收藏此色號'):
            if current_selection not in favorites:
                favorites.append(current_selection)
                save_favorites(favorites)
                st.success('已加入收藏！')
        
        # 顯示收藏列表
        if favorites:
            st.subheader('💕 我的收藏')
            for fav in favorites:
                if st.button(f'🗑️ {fav}', key=fav):
                    favorites.remove(fav)
                    save_favorites(favorites)
                    st.rerun()
        
        # 新增質地選擇和唇形修容
        texture = st.selectbox('選擇口紅質地', ['霧面', '珠光', '絲絨'])
        texture_map = {'霧面': 'matte', '珠光': 'pearl', '絲絨': 'velvet'}
        
        st.write('唇形修容')
        col1, col2 = st.columns(2)
        with col1:
            scale_x = st.slider('調整唇寬', 0.8, 1.2, 1.0, 0.05)
        with col2:
            scale_y = st.slider('調整唇高', 0.8, 1.2, 1.0, 0.05)

        # 顯示當前選擇的顏色預覽
        color_preview = np.full((50, 200, 3), LIPSTICK_COLORS[selected_brand][selected_color], dtype=np.uint8)
        st.image(color_preview, caption=f'{selected_brand} - {selected_color}')

        all_lips_points = detect_lips(image, face_mesh)
        if all_lips_points is not None:
            result = image.copy()
            for lips_points in all_lips_points:
                result = apply_lipstick(result, lips_points, 
                                       LIPSTICK_COLORS[selected_brand][selected_color],
                                       intensity, texture_map[texture],
                                       (scale_x, scale_y))
            st.image(result, caption='試妝結果')
        else:
            st.error('未偵測到唇部，請嘗試其他照片或調整姿勢／光線')

if __name__ == '__main__':
    main()
