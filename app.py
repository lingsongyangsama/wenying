import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import os
import urllib.request
from matplotlib import font_manager
import plotly.graph_objects as go

# ================= 网页基础配置 =================
st.set_page_config(page_title="超声波物理特征数字分析平台", layout="wide")
st.title("🌊 超声波干涉与传播空间高级分析系统")
st.markdown("基于单镜离轴纹影系统与机器视觉，探究超声波的波阵面、声速及能量耗散规律。")

# ================= 字体乱码终极修复 =================
@st.cache_resource
def load_chinese_font():
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
        urllib.request.urlretrieve(url, font_path)
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

load_chinese_font()

# 侧边栏：参数设置
st.sidebar.header("⚙️ 实验参数设置")
real_diameter_mm = st.sidebar.number_input("凹面镜视场真实直径 (mm)", value=203.0, step=1.0)
frequency_hz = st.sidebar.number_input("超声波发射频率 (Hz)", value=40000.0, step=100.0)

# ================= 新增：原理介绍与光路仿真 =================
st.markdown("---")
st.header("🔬 纹影成像原理与光路仿真")

col_text, col_sim = st.columns([1, 1.2])

with col_text:
    st.markdown("### 为什么能“看见”声波？")
    st.markdown("""
    超声波在空气中传播时，本质上是空气分子的纵向振动，这会导致空间中产生周期性的**疏密变化**。
    
    根据格拉德斯通-戴尔定律 (Gladstone-Dale relation)，气体的折射率 $n$ 与其密度 $\\rho$ 成正比：
    $$n - 1 = K \\rho$$
    
    当平行光束穿过测试区域时，由于超声波波阵面（波腹与波节）处的折射率 $n$ 不同，光线会发生微小的偏折（折射）。
    
    在**单镜离轴纹影光路**中，我们在反射光束的焦点处放置了一个**切光刀片 (Knife-edge)**。
    * **未偏折的光线**：被刀片精准阻挡（或部分阻挡）。
    * **偏折的光线**：若向上偏折则越过刀片，在相机中形成**亮区**；若向下偏折则被刀片完全遮挡，形成**暗区**。
    
    由此，我们将肉眼无法看见的**相位差**（折射率变化），完美转化为了相机可以捕捉的**振幅差**（光强明暗变化）。
    """)

with col_sim:
    st.markdown("### 单镜离轴纹影光路交互演示")
    # 构建交互式 Plotly 光路图
    fig_optics = go.Figure()

    # 1. 绘制凹面反射镜 (右侧圆弧)
    theta = np.linspace(-0.5, 0.5, 50)
    R = 200
    mirror_x = 200 - 20 * np.cos(theta)
    mirror_y = R * np.sin(theta)
    fig_optics.add_trace(go.Scatter(x=mirror_x, y=mirror_y, mode='lines', line=dict(color='lightblue', width=6), name='凹面反射镜'))

    # 2. 绘制光轴与对称轴
    fig_optics.add_trace(go.Scatter(x=[-250, 200], y=[0, 0], mode='lines', line=dict(color='gray', width=1, dash='dash'), showlegend=False))
    
    # 3. 绘制光源与刀片位置 (离轴)
    source_x, source_y = -200, 30
    knife_x, knife_y = -200, -30
    fig_optics.add_trace(go.Scatter(x=[source_x], y=[source_y], mode='markers', marker=dict(color='red', size=12), name='点光源'))
    fig_optics.add_trace(go.Scatter(x=[-250, -180], y=[knife_y-10, knife_y-10], mode='lines', line=dict(color='black', width=8), name='切光刀片/相机'))

    # 4. 绘制光路 (发散 -> 反射 -> 汇聚)
    rays_y = [80, 0, -80] # 打在镜面上的三个点
    for ry in rays_y:
        rx = 200 - 20 * np.cos(np.arcsin(ry/R))
        # 入射光
        fig_optics.add_trace(go.Scatter(x=[source_x, rx], y=[source_y, ry], mode='lines', line=dict(color='cyan', width=1.5), showlegend=False))
        # 反射光
        fig_optics.add_trace(go.Scatter(x=[rx, knife_x], y=[ry, knife_y], mode='lines', line=dict(color='blue', width=1.5), showlegend=False))

    # 5. 测试区域框
    fig_optics.add_shape(type="rect", x0=0, y0=-90, x1=120, y1=90, line=dict(color="gray", dash="dash"), fillcolor="rgba(0,0,0,0)")
    fig_optics.add_annotation(x=60, y=100, text="测试区域 (超声波场)", showarrow=False)

    fig_optics.update_layout(
        xaxis=dict(visible=False, range=[-250, 220]),
        yaxis=dict(visible=False, range=[-120, 120]),
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_optics, use_container_width=True)

st.markdown("---")

# ================= 核心物理模型定义 =================
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
st.header("📂 实验数据上传与解析")
uploaded_file = st.file_uploader("请在此处上传实验截图 (支持 JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.success("图像加载成功！正在进行深度物理特征解析...")
    
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
    
    st.subheader("📊 核心数据空间提取与拟合")
    fig1, axs1 = plt.subplots(2, 2, figsize=(16, 12))

    axs1[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs1[0, 0].plot(x_axis, [center_y]*len(x_axis), 'r-', alpha=0.5, label='主采样线')
    axs1[0, 0].scatter(peaks_center, [center_y]*len(peaks_center), c='red', s=10, zorder=5)
    axs1[0, 0].set_title('图1: 三线空间采样与波阵面提取', fontsize=14, fontweight='bold')
    axs1[0, 0].axis('off')

    axs1[0, 1].plot(x_axis, profile_center, 'k-', linewidth=1.5)
    axs1[0, 1].scatter(peaks_center, peak_intensities, color='red', marker='X', s=80)
    axs1[0, 1].set_title(f'图2: 测量结果 (波长 λ = {wavelength_mm:.2f} mm, 声速 v = {sound_speed_m_s:.2f} m/s)', fontsize=14, fontweight='bold')
    axs1[0, 1].grid(True, linestyle='--', alpha=0.6)

    axs1[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs1[1, 0].plot(source_x, source_y, 'r*', markersize=15, label='虚拟点声源')
    for px in peaks_center[::2]: 
        radius = np.sqrt((px - source_x)**2 + (center_y - source_y)**2)
        circle = plt.Circle((source_x, source_y), radius, color='yellow', fill=False, linestyle=':', linewidth=1.5)
        axs1[1, 0].add_patch(circle)
    axs1[1, 0].set_title('图3: 二维同心圆反向声源定位', fontsize=14, fontweight='bold')
    axs1[1, 0].axis('off')

    axs1[1, 1].scatter(peak_radial_distances, peak_intensities, color='blue', s=50)
    if fit_success:
        r_smooth = np.linspace(np.min(peak_radial_distances), np.max(peak_radial_distances), 100)
        axs1[1, 1].plot(r_smooth, realistic_decay(r_smooth, *popt), 'r-', linewidth=2.5, label=r'综合衰减模型')
    axs1[1, 1].set_title('图4: 超声波能量衰减物理分析', fontsize=14, fontweight='bold')
    axs1[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs1[1, 1].legend()
    
    plt.tight_layout()
    st.pyplot(fig1)
    
    # --- 高阶教学可视化 (3D & FFT) ---
    st.markdown("---")
    st.subheader("🎓 高阶教学可视化：打破维度限制")
    
    roi_width = int(w * 0.4)
    roi_height = int(w * 0.4)
    roi_y_start = max(0, center_y - roi_height // 2)
    roi_y_end = min(gray.shape[0], center_y + roi_height // 2)
    roi_x_start = max(0, line_start_x)
    roi_x_end = min(gray.shape[1], line_start_x + roi_width)
    roi_gray = enhanced_gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    col_3d, col_fft = st.columns(2)

    with col_3d:
        st.markdown("**教学可视化一：三维超声波声压场地形图**")
        st.caption("👈 *提示：鼠标按住图表可任意拖拽旋转，滚轮可缩放*")
        sub_sample = 4
        roi_sub = roi_gray[::sub_sample, ::sub_sample]
        X = np.arange(0, roi_sub.shape[1])
        Y = np.arange(0, roi_sub.shape[0])
        Z = roi_sub.astype(float) - np.mean(roi_sub)
        
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='RdBu_r')])
        fig_3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis_title='X 像素',
                yaxis_title='Y 像素',
                zaxis_title='声压差起伏',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=500
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_fft:
        st.markdown("**教学可视化二：二维傅里叶空间频谱图**")
        st.caption("证明空间波纹高度周期性的频域“指纹”")
        f = np.fft.fft2(roi_gray)
        fshift = np.fft.fftshift(f) 
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        fig_fft, ax_fft = plt.subplots(figsize=(6, 5))
        ax_fft.imshow(magnitude_spectrum, cmap='magma')
        ax_fft.axis('off')
        center_fft_y, center_fft_x = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
        ax_fft.axhline(center_fft_y, color='white', linestyle='--', alpha=0.3)
        ax_fft.axvline(center_fft_x, color='white', linestyle='--', alpha=0.3)
        st.pyplot(fig_fft)

    # 核心数据展示
    st.markdown("---")
    st.subheader("💡 定量计算结果")
    col1, col2 = st.columns(2)
    col1.metric("实测超声波波长 (λ)", f"{wavelength_mm:.2f} mm")
    col2.metric("推算空气声速 (v)", f"{sound_speed_m_s:.2f} m/s")

else:
    st.info("💡 理论与光路已就绪。请在上方上传实际拍摄的纹影图像，系统将自动执行解析。")
