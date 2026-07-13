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
from fpdf import FPDF  

# ================= 网页基础配置 =================
st.set_page_config(page_title="超声波物理特征数字分析平台", layout="wide")
st.title("🌊 超声波干涉与传播空间高级分析系统")
st.markdown("基于单镜离轴纹影系统与机器视觉，探究超声波的波阵面、声速及能量耗散规律。")

# ================= 字体乱码终极修复 =================
@st.cache_resource
def load_chinese_font():
    font_path = "SimHei.ttf"
    if not os.path.exists(font_path):
        try:
            url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
            urllib.request.urlretrieve(url, font_path)
            font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['SimHei']
        except Exception as e:
            st.warning("⚠️ 中文字体下载失败，图表可能出现乱码。请检查网络或在本地放置 SimHei.ttf。")
    else:
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

load_chinese_font()

# ================= 侧边栏：参数设置 =================
st.sidebar.header("⚙️ 实验参数设置")
real_diameter_mm = st.sidebar.number_input("凹面镜视场真实直径 (mm)", value=203.0, step=1.0)
frequency_hz = st.sidebar.number_input("超声波发射频率 (Hz)", value=40000.0, step=100.0)

# 高级视觉参数（带通俗说明悬停提示）
with st.sidebar.expander("🔧 高级视觉参数 (建议默认)"):
    threshold_val = st.number_input(
        "视场二值化阈值", min_value=0, max_value=255, value=30, step=1,
        help="【找圆圈用的】调节这个数字，能帮电脑把‘圆形的镜面’从‘黑色的背景’中完整地抠出来。如果系统报错说没找到圆，试着微调一下它。"
    )
    offset_val = st.number_input(
        "辅助采样线偏移量 (像素)", min_value=10, max_value=200, value=40, step=5,
        help="【找声源用的】除了中间那条主线，我们还在上下各画了一条线来辅助定位。这个数值决定了上下两条线离中间有多远。"
    )
    sg_window = st.number_input(
        "平滑滤波窗口大小", min_value=5, max_value=101, value=31, step=2,
        help="【去噪点用的】照片上可能会有杂乱的‘雪花’。这个数字越大，电脑画出的波浪线就越平滑；但如果调得太大，可能会把真实的声波细节给抹平哦。"
    )
    if sg_window % 2 == 0:
        sg_window += 1

# ================= 原理介绍与光路仿真 =================
st.markdown("---")
with st.expander("🔬 纹影成像原理与光路仿真 (点击展开/折叠)", expanded=False):
    st.markdown("### 为什么能“看见”声波？")
    st.markdown("""
    超声波在空气中传播时，本质上是空气分子的纵向振动，这会导致空间中产生周期性的**疏密变化**。

    根据格拉德斯通-戴尔定律 (Gladstone-Dale relation)，气体的折射率 $n$ 与其密度 $\\rho$ 成正比：
    $$n - 1 = K \\rho$$

    当平行光束穿过测试区域时，由于超声波波阵面（波腹与波节）处的折射率 $n$ 不同，光线会发生微小的偏折（折射）。

    在**单镜离轴纹影光路**中，我们在反射光束的焦点处放置了一个**切光刀片 (Knife-edge)**：
    * **未偏折的光线**：被刀片精准阻挡（或部分阻挡）。
    * **偏折的光线**：若向上偏折则越过刀片，在相机中形成**亮区**；若向下偏折则被刀片完全遮挡，形成**暗区**。

    由此，我们将肉眼无法看见的**相位差**（折射率变化），完美转化为了相机可以捕捉的**振幅差**（光强明暗变化）。
    """)

    st.markdown("### 单镜离轴纹影光路示意图")
    fig_optics = go.Figure()

    fig_optics.update_xaxes(visible=False, range=[-320, 260], showgrid=False, zeroline=False)
    fig_optics.update_yaxes(visible=False, range=[-130, 130], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)

    source_x, source_y = -180, 40
    knife_x, knife_y = -180, -40
    mirror_r = 250
    center_of_curvature = 200 - mirror_r 

    fig_optics.add_trace(go.Scatter(x=[-300, 240], y=[0, 0], mode='lines', line=dict(color='#CBD5E1', width=2, dash='dash'), hoverinfo='skip'))
    fig_optics.add_annotation(x=250, y=0, text="光轴", showarrow=False, font=dict(color="gray"))

    fig_optics.add_trace(go.Scatter(x=[source_x, -180, knife_x], y=[source_y, 0, knife_y], mode='lines', line=dict(color='purple', width=2, dash='dot'), hoverinfo='skip'))
    fig_optics.add_annotation(x=-120, y=0, text="关于光轴离轴对称", showarrow=False, font=dict(color="purple", size=12))

    fig_optics.add_shape(type="rect", x0=0, y0=-90, x1=150, y1=90, line=dict(color="#94A3B8", dash="dash", width=2), fillcolor="rgba(0,0,0,0)")
    fig_optics.add_annotation(x=75, y=105, text="<b>测试区域</b>", showarrow=False, font=dict(size=14))
    fig_optics.add_annotation(x=75, y=-105, text="紧靠反射镜正前方", showarrow=False, font=dict(size=10, color="gray"))

    rays_y = [80, 0, -80] 
    for ry in rays_y:
        rx = center_of_curvature + mirror_r * np.cos(np.arcsin(ry/mirror_r))
        fig_optics.add_trace(go.Scatter(x=[source_x, rx], y=[source_y, ry], mode='lines', line=dict(color='#60A5FA', width=1.5), hoverinfo='skip'))
        fig_optics.add_trace(go.Scatter(x=[rx, knife_x], y=[ry, knife_y], mode='lines', line=dict(color='#2563EB', width=1.5), hoverinfo='skip'))
        fig_optics.add_trace(go.Scatter(x=[knife_x, knife_x - 70], y=[knife_y, knife_y - (ry-knife_y)*(70/(rx-knife_x))], mode='lines', line=dict(color='#2563EB', width=1.5), hoverinfo='skip'))

    theta = np.linspace(-0.45, 0.45, 50)
    mirror_x = center_of_curvature + mirror_r * np.cos(theta)
    mirror_y = mirror_r * np.sin(theta)
    fig_optics.add_trace(go.Scatter(x=mirror_x, y=mirror_y, mode='lines', line=dict(color='#93C5FD', width=8), hoverinfo='skip'))
    fig_optics.add_trace(go.Scatter(x=[200], y=[0], mode='markers', marker=dict(color='red', symbol='cross', size=10), hoverinfo='skip'))

    fig_optics.add_shape(type="rect", x0=-230, y0=25, x1=-190, y1=55, fillcolor="#334155", line_width=0, layer="below") 
    fig_optics.add_shape(type="rect", x0=-200, y0=-100, x1=-185, y1=-42, fillcolor="#1E293B", line_width=0, layer="below") 
    fig_optics.add_shape(type="rect", x0=-290, y0=-60, x1=-250, y1=-20, fillcolor="#1E293B", line_width=0, layer="below")
    fig_optics.add_shape(type="path", path=f"M -250 -30 L -230 -20 L -230 -60 L -250 -50 Z", fillcolor="#94A3B8", line_width=0, layer="below")

    fig_optics.add_trace(go.Scatter(x=[source_x, knife_x], y=[source_y, knife_y], mode='markers', marker=dict(color='red', size=12, line=dict(color='rgba(255,0,0,0.3)', width=4)), hoverinfo='skip'))

    fig_optics.add_annotation(x=-180, y=70, text="<b>点光源</b><br><span style='font-size:10px; color:gray'>光源出口</span>", showarrow=False, align="left")
    fig_optics.add_annotation(x=-150, y=-60, text="<b>切光刀片</b><br><span style='font-size:10px; color:gray'>置于反射像点处</span>", showarrow=False, align="left")
    fig_optics.add_annotation(x=-270, y=-80, text="<b>相机</b>", showarrow=False)
    fig_optics.add_annotation(x=260, y=50, text="<b>凹面反射镜</b><br><span style='font-size:10px; color:gray'>镜面中心位于光轴</span>", showarrow=False, align="left")

    fig_optics.add_shape(type="line", x0=-180, y0=-120, x1=200, y1=-120, line=dict(color="#475569", width=2))
    fig_optics.add_annotation(x=-180, y=-120, text="◀", showarrow=False, font=dict(color="#475569"))
    fig_optics.add_annotation(x=200, y=-120, text="▶", showarrow=False, font=dict(color="#475569"))
    fig_optics.add_annotation(x=10, y=-120, text="<b> 光源出口至镜面中心：2f </b>", showarrow=False, bgcolor="white", bordercolor="#475569", borderwidth=1, borderpad=4, font=dict(size=12))

    fig_optics.update_layout(
        showlegend=False, dragmode=False, hovermode=False, 
        margin=dict(l=0, r=0, t=20, b=0), height=500, plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_optics, use_container_width=True, config={'displayModeBar': False})

# ================= 新增：现象观察板块 =================
st.markdown("---")
st.header("👀 现象观察")
st.markdown("通过单镜离轴纹影系统，我们可以清晰地观察到空气折射率的变化，将不可见的“热”与“声”转化为肉眼可见的震撼影像。")

tab_heat, tab_sound, tab_levitation = st.tabs(["🔥 热现象", "🌊 声波反射与干涉", "🛸 声悬浮"])

# --- 1. 热现象 ---
with tab_heat:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**热水上方的热对流**")
        if os.path.exists("rs.mp4"):
            st.video("rs.mp4")
        else:
            st.info("尚未上传视频: rs.mp4")
            
    with col2:
        st.markdown("**打火机火焰与热气流**")
        if os.path.exists("hy.mp4"):
            st.video("hy.mp4")
        else:
            st.info("尚未上传视频: hy.mp4")

# --- 2. 声波反射与干涉 ---
with tab_sound:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🪞 声波反射")
        st.markdown("超声波遇到刚性边界时发生反射，入射波与反射波叠加。")
        if os.path.exists("fs.png"):
            st.image("fs.png", caption="声波反射静态图", use_column_width=True)
        else:
            st.info("尚未上传图片: fs.png")
            
        if os.path.exists("fs.mp4"):
            st.video("fs.mp4")
        else:
            st.info("尚未上传视频: fs.mp4")
            
    with col2:
        st.markdown("### 🧬 声波干涉")
        st.markdown("多个声源或入射波与反射波在空间中相遇，形成稳定的相干条纹。")
        if os.path.exists("gs.mp4"):
            st.video("gs.mp4")
        else:
            st.info("尚未上传视频: gs.mp4")

# --- 3. 声悬浮 ---
with tab_levitation:
    st.markdown("### 🛸 超声波悬浮现象")
    st.markdown("利用发射端与反射端之间形成的**驻波**，将轻小物体（如泡沫球）精准稳定在声压节点处，成功克服重力实现悬浮。")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if os.path.exists("sxf.mp4"):
            st.video("sxf.mp4")
        else:
            st.info("尚未上传视频: sxf.mp4")
    with col2:
        if os.path.exists("yg.mp4"):
            st.video("yg.mp4")
        else:
            st.info("尚未上传视频: yg.mp4")
    with col3:
        if os.path.exists("lg.mp4"):
            st.video("lg.mp4")
        else:
            st.info("尚未上传视频: lg.mp4")

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

# ================= 文件上传与示例模块 =================
st.markdown("---")
st.header("📂 实验数据上传与解析")

if os.path.exists("example.jpg"):
    with open("example.jpg", "rb") as file:
        st.download_button(
            label="📥 点击下载示例纹影图像用于测试",
            data=file,
            file_name="example.jpg",
            mime="image/jpeg"
        )
else:
    st.info("💡 提示：您可以将上课拍摄的清晰图片重命名为 `example.jpg` 并存放在代码同级目录下，系统会自动在此处生成供学生下载的按钮。")

uploaded_file = st.file_uploader("请在此处上传实验截图 (支持 JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    with st.spinner("正在进行深度物理特征解析，请稍候..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.success("图像加载成功！数据计算已完成。")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)

        # 1. 物理比例尺
        _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            mm_per_pixel = real_diameter_mm / max(w, h)
        else:
            st.error("无法识别圆形视场，请调大/调小左侧栏的【视场二值化阈值】，或检查图像清晰度。")
            st.stop()

        # 2. 空间提取
        center_y = y + h // 2
        line_start_x = x + int(w * 0.15)
        line_end_x = x + int(w * 0.85)

        def extract_and_find_peaks(y_coord):
            y_safe = np.clip(y_coord, 5, enhanced_gray.shape[0] - 6)
            profile = np.mean(enhanced_gray[y_safe - 5 : y_safe + 5, line_start_x:line_end_x], axis=0)
            smooth = savgol_filter(profile, window_length=int(sg_window), polyorder=3)
            peaks, _ = find_peaks(smooth, distance=(w//30)*0.5, prominence=3)
            return smooth, peaks + line_start_x

        profile_center, peaks_center = extract_and_find_peaks(center_y)
        _, peaks_top = extract_and_find_peaks(center_y - offset_val)
        _, peaks_bottom = extract_and_find_peaks(center_y + offset_val)

        # 3. 声源反推
        calculated_centers_x = []
        calculated_centers_y = []
        for px in peaks_center:
            top_match = [p for p in peaks_top if abs(p - px) < 20]
            bottom_match = [p for p in peaks_bottom if abs(p - px) < 20]
            if top_match and bottom_match:
                pt_center = (px, center_y)
                pt_top = (top_match[0], center_y - offset_val)
                pt_bottom = (bottom_match[0], center_y + offset_val)
                center = calc_circle_center(pt_top, pt_center, pt_bottom)
                # 修复了这里的逻辑判定：X坐标应当与0比较，而不是与Y坐标比较
                if center and center[0] > 0 and center[0] < image.shape[1] + 500:
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
        except RuntimeError: 
            fit_success = False
        except Exception:
            fit_success = False

    # ================= 渲染网页图表 =================
    st.subheader("📊 核心数据空间提取与拟合")
    
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.markdown("**图1: 三线空间采样与波阵面提取**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.plot(x_axis, [center_y]*len(x_axis), 'r-', alpha=0.5, label='主采样线')
        ax1.plot(x_axis, [center_y - offset_val]*len(x_axis), 'g--', alpha=0.3, label='上采样线')
        ax1.plot(x_axis, [center_y + offset_val]*len(x_axis), 'g--', alpha=0.3, label='下采样线')
        ax1.scatter(peaks_center, [center_y]*len(peaks_center), c='red', s=15, zorder=5)
        ax1.axis('off')
        fig1.tight_layout(pad=0)
        st.pyplot(fig1)
        with st.expander("💡 图1在干嘛？"):
            st.markdown("你看照片上的明暗条纹，那就是超声波！电脑在画面上‘拉’了三条横线，去感受哪里最亮（波腹）。**红点**就是电脑准确抓到的每一个声波波峰的位置。")

    with row1_col2:
        st.markdown(f"**图2: 测量结果**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_axis, y=profile_center, mode='lines', name='灰度剖面', line=dict(color='black', width=1.5)))
        fig2.add_trace(go.Scatter(x=peaks_center, y=peak_intensities, mode='markers', name='提取峰值', marker=dict(color='red', symbol='x', size=8)))
        fig2.update_layout(
            xaxis_title="像素 X 坐标", yaxis_title="光强 (灰度)",
            margin=dict(l=20, r=20, t=10, b=20),
            height=350,
            hovermode="x unified"
        )
        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("💡 图2怎么看？"):
            st.markdown("这是把图1中间那条线的光强变化‘画’成了波浪线。两个红色叉叉之间的距离，在物理上就代表了一个**波长**！利用波长和已知的频率，我们就能算出声音传播的速度了。")

    with row2_col1:
        st.markdown("**图3: 二维同心圆反向声源定位**")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax3.plot(source_x, source_y, 'r*', markersize=15, label='虚拟点声源')
        for px in peaks_center[::2]: 
            radius = np.sqrt((px - source_x)**2 + (center_y - source_y)**2)
            circle = plt.Circle((source_x, source_y), radius, color='yellow', fill=False, linestyle=':', linewidth=1.5)
            ax3.add_patch(circle)
        ax3.axis('off')
        fig3.tight_layout(pad=0)
        st.pyplot(fig3)
        with st.expander("💡 图3的同心圆代表什么？"):
            st.markdown("想象一下往水池里扔一颗石子，波纹是一圈圈扩散的。根据图1里上下中三个红点的位置，电脑利用几何知识（三点确定一个圆），像侦探一样**反向推算**出了发射超声波的探头（红星位置）到底藏在画面外面的哪里！")

    with row2_col2:
        st.markdown("**图4: 超声波能量衰减物理分析**")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=peak_radial_distances, y=peak_intensities, mode='markers', name='实测峰值', marker=dict(color='blue', size=8)))
        if fit_success:
            r_smooth = np.linspace(np.min(peak_radial_distances), np.max(peak_radial_distances), 100)
            fig4.add_trace(go.Scatter(x=r_smooth, y=realistic_decay(r_smooth, *popt), mode='lines', name='综合衰减模型', line=dict(color='red', width=2)))
        fig4.update_layout(
            xaxis_title="距声源径向距离 (像素)", yaxis_title="波峰相对光强",
            margin=dict(l=20, r=20, t=10, b=20),
            height=350,
            hovermode="closest"
        )
        st.plotly_chart(fig4, use_container_width=True)
        with st.expander("💡 声音是怎么变弱的？"):
            st.markdown("常识告诉我们，离得越远，声音越小。图上的蓝点是我们真实测到的声音能量，红线是物理学家通过数学公式算出来的理论衰减曲线。你可以看看，我们实测的数据跟科学家的理论吻合得漂不漂亮！")
    
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
            height=400
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_fft:
        st.markdown("**教学可视化二：二维傅里叶空间频谱图**")
        st.caption("证明空间波纹高度周期性的频域“指纹”")
        f = np.fft.fft2(roi_gray)
        fshift = np.fft.fftshift(f) 
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        fig_fft, ax_fft = plt.subplots(figsize=(5, 4))
        ax_fft.imshow(magnitude_spectrum, cmap='magma')
        ax_fft.axis('off')
        center_fft_y, center_fft_x = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
        ax_fft.axhline(center_fft_y, color='white', linestyle='--', alpha=0.3)
        ax_fft.axvline(center_fft_x, color='white', linestyle='--', alpha=0.3)
        st.pyplot(fig_fft)

    # ================= 互动教学：读图计算与结果核对 =================
    st.markdown("---")
    st.subheader("🧠 探究挑战：根据图像自己算出声速！")
    st.markdown("利用上面的**图2**，你能自己算出空气中的声速吗？")

    col_guide, col_calc = st.columns([1.2, 1])

    with col_guide:
        st.info(f"**📝 计算指南与已知条件**\n\n"
                f"1. **求像素距离**：把鼠标悬停在【图2】的波峰（红叉）上，读出相邻两个波峰的 X 坐标并相减，这就是一个波长包含的像素数。*(💡提示：为了减小误差，你可以读取相隔5个波峰的距离，再除以5！)*\n\n"
                f"2. **换算物理波长 (λ)**：系统通过测量镜面轮廓，算出了当前照片的物理比例尺为 **1 像素 = {mm_per_pixel:.4f} mm**。将上一步的像素距离乘以它，得到实际波长。\n\n"
                f"3. **计算声速 (v)**：已知超声波探头的发射频率 f = **{frequency_hz} Hz**。利用波速公式 $v = \lambda \\times f$ 即可求出声速。*(⚠️记得把 mm 换算成 m 喔！)*")

    with col_calc:
        st.markdown("**✏️ 填入你的计算结果**")
        student_lambda = st.number_input("你算出的波长 λ (mm)", min_value=0.0, value=0.0, step=0.1, format="%.2f")
        student_v = st.number_input("你算出的声速 v (m/s)", min_value=0.0, value=0.0, step=0.1, format="%.2f")

    with st.expander("👀 算完了吗？点击这里核对系统的精准分析结果！", expanded=False):
        st.markdown("系统提取了主线上所有的波峰数据进行了综合平均运算，得到了当前的精准数值：")
        
        res_col1, res_col2 = st.columns(2)
        
        delta_lambda = f"差值 {student_lambda - wavelength_mm:.2f} mm" if student_lambda > 0 else None
        delta_v = f"差值 {student_v - sound_speed_m_s:.2f} m/s" if student_v > 0 else None
        
        res_col1.metric("系统实测超声波波长 (λ)", f"{wavelength_mm:.2f} mm", delta=delta_lambda, delta_color="off")
        res_col2.metric("系统推断空气声速 (v)", f"{sound_speed_m_s:.2f} m/s", delta=delta_v, delta_color="off")
        
        # 定义初始状态和报告所需的评语变量
        error_percent = 0.0
        eval_text = "（学生尚未提交自主计算结果进行对比评估）"
        
        if student_v > 0:
            st.markdown("### 🎯 你的误差智能分析")
            error_percent = abs(student_v - sound_speed_m_s) / sound_speed_m_s * 100
            
            if error_percent < 3:
                eval_text = "你读取的数据非常精准，而且完美避开了单位换算的陷阱，具备了严谨的科学素养！"
                st.success(f"🎉 **完美！相对误差仅为 {error_percent:.2f}%！** \n\n{eval_text}")
            elif student_v > 10000: 
                eval_text = "你算出来的声速比火箭还要快！仔细看看你的计算过程，是不是忘记把波长的单位从毫米 (mm) 换算成米 (m) 就直接跟频率相乘了？回去改一下试试！"
                st.error(f"😱 **相对误差极大！** \n\n{eval_text}")
            else:
                eval_text = "单位换算应该是对的，但取点可能不够准。在图2中读取相隔较远（比如第1个和第8个）的红叉的 X 坐标，相减后除以中间包含的波段数，这样能极大减小偶然误差！"
                st.warning(f"🤔 **相对误差为 {error_percent:.2f}%。大方向对了，但有一点小偏差哦！** \n\n{eval_text}")

        st.caption("注：系统采用了全像素阵列多点均值技术，因此可能会与你手动选取两点计算的结果有微小差异，这是正常的实验误差。")
        
        # ================= 生成与下载包含全套图表的 PDF 实验报告 =================
        st.markdown("---")
        st.markdown("### 📄 专属实验报告导出")
        st.caption("学生可以一键将当前图像的处理结果（包含完整的4组物理分析图表）、自主测算数据以及系统的智能误差评估打包为标准 PDF 报告。")
        
        # --- 后台将 4 张图表保存为本地图片 ---
        fig1.savefig("temp_fig1.png", dpi=150, bbox_inches='tight')
        fig3.savefig("temp_fig3.png", dpi=150, bbox_inches='tight')

        # 用 matplotlib 重绘图2和图4（避开Plotly在云端服务器导出时的环境变量深坑）
        fig2_pdf, ax2_pdf = plt.subplots(figsize=(6, 4))
        ax2_pdf.plot(x_axis, profile_center, 'k-', linewidth=1.5, label='灰度剖面')
        ax2_pdf.scatter(peaks_center, peak_intensities, color='red', marker='X', s=60, label='提取峰值')
        ax2_pdf.set_title(f'图2: 测量结果 (波长 λ = {wavelength_mm:.2f} mm)', fontsize=12)
        ax2_pdf.set_xlabel("像素 X 坐标")
        ax2_pdf.set_ylabel("光强 (灰度)")
        ax2_pdf.legend()
        fig2_pdf.tight_layout()
        fig2_pdf.savefig("temp_fig2.png", dpi=150, bbox_inches='tight')
        plt.close(fig2_pdf) # 释放内存

        fig4_pdf, ax4_pdf = plt.subplots(figsize=(6, 4))
        ax4_pdf.scatter(peak_radial_distances, peak_intensities, color='blue', s=40, label='实测峰值')
        if fit_success:
            r_smooth_pdf = np.linspace(np.min(peak_radial_distances), np.max(peak_radial_distances), 100)
            ax4_pdf.plot(r_smooth_pdf, realistic_decay(r_smooth_pdf, *popt), 'r-', linewidth=2, label='综合衰减模型')
        ax4_pdf.set_title('图4: 超声波能量衰减物理分析', fontsize=12)
        ax4_pdf.set_xlabel("距声源径向距离 (像素)")
        ax4_pdf.set_ylabel("波峰相对光强")
        ax4_pdf.legend()
        fig4_pdf.tight_layout()
        fig4_pdf.savefig("temp_fig4.png", dpi=150, bbox_inches='tight')
        plt.close(fig4_pdf) # 释放内存

        # --- 开始排版 PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("SimHei", "", "SimHei.ttf", uni=True) 
        
        pdf.set_font("SimHei", size=18)
        pdf.cell(0, 15, txt="超声波干涉与声速空间数字分析实验报告", ln=True, align="C")
        pdf.set_font("SimHei", size=12)
        pdf.cell(0, 8, txt="-"*65, ln=True, align="C")
        pdf.cell(0, 5, txt="", ln=True)
        
        # 第一部分：数据记录
        pdf.set_font("SimHei", size=14)
        pdf.cell(0, 10, txt="一、 系统视觉提取与核心算法基准数据", ln=True)
        pdf.set_font("SimHei", size=12)
        pdf.cell(0, 8, txt=f"  - 图像物理比例尺: 1 像素 = {mm_per_pixel:.4f} mm", ln=True)
        pdf.cell(0, 8, txt=f"  - 探头发射频率 f: {frequency_hz} Hz", ln=True)
        pdf.cell(0, 8, txt=f"  - 多点均值法测量波长 λ: {wavelength_mm:.2f} mm", ln=True)
        pdf.cell(0, 8, txt=f"  - 算法推荐理论声速 v: {sound_speed_m_s:.2f} m/s", ln=True)
        pdf.cell(0, 5, txt="", ln=True)
        
        # 第二部分：自主测算
        pdf.set_font("SimHei", size=14)
        pdf.cell(0, 10, txt="二、 学生自主读图探究结果记录", ln=True)
        pdf.set_font("SimHei", size=12)
        if student_v > 0:
            pdf.cell(0, 8, txt=f"  - 自主测算声波波长 λ: {student_lambda:.2f} mm", ln=True)
            pdf.cell(0, 8, txt=f"  - 自主推算空气声速 v: {student_v:.2f} m/s", ln=True)
            pdf.cell(0, 8, txt=f"  - 测算结果相对误差: {error_percent:.2f}%", ln=True)
        else:
            pdf.cell(0, 8, txt="  - （学生尚未在平台上输入自主测算数据）", ln=True)
        pdf.cell(0, 5, txt="", ln=True)
            
        # 第三部分：评语
        pdf.set_font("SimHei", size=14)
        pdf.cell(0, 10, txt="三、 探究表现智能评估反馈", ln=True)
        pdf.set_font("SimHei", size=12)
        pdf.multi_cell(0, 8, txt=f"  {eval_text}")
        pdf.cell(0, 5, txt="", ln=True)

        # 第四部分：四宫格附录图像
        pdf.add_page() # 新起一页，防止图文错乱
        pdf.set_font("SimHei", size=14)
        pdf.cell(0, 10, txt="四、 实验过程物理图像与数据拟合档案", ln=True)
        pdf.cell(0, 5, txt="", ln=True)

        # 把刚才后台生成的四张图片按照2x2宫格贴到 PDF 里
        y_curr = pdf.get_y()
        pdf.image("temp_fig1.png", x=10, y=y_curr, w=90)
        pdf.image("temp_fig2.png", x=105, y=y_curr, w=90)

        pdf.set_y(y_curr + 70) 
        pdf.image("temp_fig3.png", x=10, y=pdf.get_y(), w=90)
        pdf.image("temp_fig4.png", x=105, y=pdf.get_y(), w=90)
        
        pdf_bytes = bytes(pdf.output())
        
        # 用完图片后立刻清理，不占用服务器内存空间
        for f in ["temp_fig1.png", "temp_fig2.png", "temp_fig3.png", "temp_fig4.png"]:
            if os.path.exists(f):
                os.remove(f)
        
        st.download_button(
            label="⬇️ 一键生成并下载PDF实验报告",
            data=pdf_bytes,
            file_name="超声波测声速_数字分析实验报告.pdf",
            mime="application/pdf"
        )

else:
    st.info("💡 期待您的探索！请在上方上传实际拍摄的纹影图像，系统将自动执行解析。")
