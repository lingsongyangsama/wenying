import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

# ================= 网页基础配置 =================
st.set_page_config(page_title="超声波物理特征数字分析平台", layout="wide")
st.title("🌊 超声波干涉与传播空间分析系统")
st.markdown("上传纹影法拍摄的超声波干涉/传播图像，系统将自动进行波阵面提取、声速推算及能量衰减物理建模。")

# 侧边栏：参数设置
st.sidebar.header("⚙️ 实验参数设置")
real_diameter_mm = st.sidebar.number_input("凹面镜视场真实直径 (mm)", value=203.0, step=1.0)
frequency_hz = st.sidebar.number_input("超声波发射频率 (Hz)", value=40000.0, step=100.0)

# ================= 核心物理模型 =================
def realistic_decay(r, a, alpha, c):
    r_safe = np.maximum(r, 1e-5)
    return a * (1 / np.sqrt(r_safe)) * np.exp(-alpha * r_safe) + c

def calc_circle_center(p1, p2, p3):
    temp = p2[0]*p2[0] + p2[1]*p2[1]
    bc = (p1[0]*p1[0] + p1[1]*p1[1] - temp) / 2
    cd = (temp - p3[0]*p3[0] - p3[1]*p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1e-6: return None 
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0])*cd - (p2[0] - p3[0])*bc) / det
    return (cx, cy)

# ================= 文件上传模块 =================
uploaded_file = st.file_uploader("请在此处上传实验截图 (支持 JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 将上传的文件转换为 OpenCV 可读的格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.success("图像加载成功！正在进行深度物理特征解析...")
    
    # --- 以下为核心处理逻辑 (与原代码一致) ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)

    # 1. 物理比例尺
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        mm_per_pixel = real_diameter_mm / max(w, h)
    else:
        st.error("无法识别圆形视场，请检查图像清晰度。")
        st.stop()

    # 2. 空间提取
    center_y = y + h // 2
    offset = 40 
    line_start_x = x + int(w * 0.15)
    line_end_x = x + int(w * 0.85)

    def extract_and_find_peaks(y_coord):
        profile = np.mean(enhanced_gray[y_coord - 5 : y_coord + 5, line_start_x:line_end_x], axis=0)
        smooth = savgol_filter(profile, window_length=31, polyorder=3)
        peaks, _ = find_peaks(smooth, distance=(w//30)*0.5, prominence=3)
        return smooth, peaks + line_start_x

    profile_center, peaks_center = extract_and_find_peaks(center_y)
    _, peaks_top = extract_and_find_peaks(center_y - offset)
    _, peaks_bottom = extract_and_find_peaks(center_y + offset)

    # 3. 声源反推
    calculated_centers_x = []
    calculated_centers_y = []
    for px in peaks_center:
        top_match = [p for p in peaks_top if abs(p - px) < 20]
        bottom_match = [p for p in peaks_bottom if abs(p - px) < 20]
        if top_match and bottom_match:
            pt_center = (px, center_y)
            pt_top = (top_match[0], center_y - offset)
            pt_bottom = (bottom_match[0], center_y + offset)
            center = calc_circle_center(pt_top, pt_center, pt_bottom)
            if center and center[0] > center_y and center[0] < image.shape[1] + 500:
                calculated_centers_x.append(center[0])
                calculated_centers_y.append(center[1])
                
    source_x = np.median(calculated_centers_x) if calculated_centers_x else image.shape[1]
    source_y = np.median(calculated_centers_y) if calculated_centers_y else center_y

    # 4. 波长与声速
    if len(peaks_center) > 1:
        radial_distances = np.sqrt((peaks_center - source_x)**2 + (center_y - source_y)**2)
        avg_pixel_dist = np.mean(np.abs(np.diff(radial_distances)))
        wavelength_mm = avg_pixel_dist * mm_per_pixel
        sound_speed_m_s = frequency_hz * (wavelength_mm / 1000.0)
    else:
        wavelength_mm, sound_speed_m_s = 0, 0

    # 5. 衰减拟合
    x_axis = np.arange(line_start_x, line_end_x)
    peak_x_coords = peaks_center
    peak_intensities = profile_center[peaks_center - line_start_x]
    peak_radial_distances = np.sqrt((peak_x_coords - source_x)**2 + (center_y - source_y)**2)
    
    try:
        p0 = [np.max(peak_intensities) * np.sqrt(np.min(peak_radial_distances)), 0.001, np.min(peak_intensities)]
        popt, _ = curve_fit(realistic_decay, peak_radial_distances, peak_intensities, p0=p0, maxfev=5000)
        fit_success = True
    except:
        fit_success = False

    # ================= 渲染网页图表 =================
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 图1
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].plot(x_axis, [center_y]*len(x_axis), 'r-', alpha=0.5, label='主采样线')
    axs[0, 0].scatter(peaks_center, [center_y]*len(peaks_center), c='red', s=10, zorder=5)
    axs[0, 0].set_title('图1: 三线空间采样与波阵面提取', fontsize=14, fontweight='bold')
    axs[0, 0].axis('off')

    # 图2
    axs[0, 1].plot(x_axis, profile_center, 'k-', linewidth=1.5)
    axs[0, 1].scatter(peaks_center, peak_intensities, color='red', marker='X', s=80)
    axs[0, 1].set_title(f'图2: 测量结果 (波长 $\lambda$ = {wavelength_mm:.2f} mm, 声速 $v$ = {sound_speed_m_s:.2f} m/s)', fontsize=14, fontweight='bold')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 图3
    axs[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[1, 0].plot(source_x, source_y, 'r*', markersize=15, label='虚拟点声源')
    for px in peaks_center[::2]: 
        radius = np.sqrt((px - source_x)**2 + (center_y - source_y)**2)
        circle = plt.Circle((source_x, source_y), radius, color='yellow', fill=False, linestyle=':', linewidth=1.5)
        axs[1, 0].add_patch(circle)
    axs[1, 0].set_title('图3: 二维同心圆反向声源定位', fontsize=14, fontweight='bold')
    axs[1, 0].axis('off')

    # 图4
    axs[1, 1].scatter(peak_radial_distances, peak_intensities, color='blue', s=50)
    if fit_success:
        r_smooth = np.linspace(np.min(peak_radial_distances), np.max(peak_radial_distances), 100)
        axs[1, 1].plot(r_smooth, realistic_decay(r_smooth, *popt), 'r-', linewidth=2.5, label=r'综合衰减模型: $A \cdot \frac{1}{\sqrt{r}} e^{-\alpha r} + C$')
    axs[1, 1].set_title('图4: 超声波能量衰减物理分析', fontsize=14, fontweight='bold')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].legend()
    
    plt.tight_layout()
    
    # 将生成的 Matplotlib 图表展示在网页上
    st.pyplot(fig)
    
    # 也可以将核心数据以大字号单独展示在网页下方，更具视觉冲击力
    st.markdown("---")
    st.subheader("💡 核心数据提取结果")
    col1, col2 = st.columns(2)
    col1.metric("实测超声波波长", f"{wavelength_mm:.2f} mm")
    col2.metric("推算空气声速", f"{sound_speed_m_s:.2f} m/s")

else:
    st.info("💡 请在上方上传图片，系统将自动开始计算。")
