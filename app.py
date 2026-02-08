import streamlit as st
from shapely.wkt import loads
from shapely.geometry import LineString, MultiLineString
from shapely.affinity import translate, scale
import folium
from folium.plugins import SideBySideLayers
from streamlit_folium import st_folium
import re
import pandas as pd
import altair as alt
import streamlit.components.v1 as components
import google.generativeai as genai

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Axes Systems: Spatial Intelligence", 
    layout="wide", 
    page_icon="🗺️",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    div[data-testid="stMetric"] {
        background-color: #1F2937; 
        padding: 15px; 
        border-radius: 10px;
        border-left: 5px solid #00C9FF; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #E0E0E0; }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: black;
        font-weight: bold;
        border: none;
        transition: transform 0.2s;
    }
    .stButton>button:hover { transform: scale(1.05); }
    .badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 14px;
        margin-right: 5px;
        color: white;
        display: inline-block;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE INIT ---
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False
if "ai_report_text" not in st.session_state:
    st.session_state["ai_report_text"] = ""

# --- 4. HELPER FUNCTIONS ---

@st.cache_data
def parse_wkt_data(raw_text):
    geometries = []
    if not raw_text or not raw_text.strip(): return geometries
    try:
        clean_text = raw_text.replace('\n', ' ').replace('\r', ' ')
        matches = re.findall(r'LINESTRING\s*\([^)]+\)', clean_text, re.IGNORECASE)
        for wkt in matches: geometries.append(loads(wkt))
    except Exception as e: st.error(f"Error parsing WKT: {e}")
    return geometries

def normalize_to_target(geometries, target_lat, target_lon):
    if not geometries: return []
    try:
        min_x = min(g.bounds[0] for g in geometries)
        min_y = min(g.bounds[1] for g in geometries)
        max_x = max(g.bounds[2] for g in geometries)
        max_y = max(g.bounds[3] for g in geometries)
    except: return geometries 
    
    width = max_x - min_x
    height = max_y - min_y
    if width == 0: width = 1
    if height == 0: height = 1

    scale_factor = 0.00001 
    final_geoms = []
    for g in geometries:
        shifted = translate(g, xoff=-min_x - (width/2), yoff=-min_y - (height/2))
        scaled = scale(shifted, xfact=scale_factor, yfact=scale_factor, origin=(0,0))
        moved = translate(scaled, xoff=target_lon, yoff=target_lat)
        final_geoms.append(moved)
    return final_geoms

def add_to_map(geom, group, color, weight, tooltip):
    if geom.is_empty: return
    parts = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    for part in parts:
        coords = [(p[1], p[0]) for p in part.coords]
        folium.PolyLine(
            coords, color=color, weight=weight, opacity=0.8, tooltip=tooltip
        ).add_to(group)

def calculate_metrics(highway, roads, displaced_roads, safe_dist):
    h_buffer = highway.buffer(safe_dist)
    tp, fp, fn, tn = 0, 0, 0, 0
    for orig, new in zip(roads, displaced_roads):
        was_unsafe = orig.intersects(h_buffer)
        is_moved = orig != new
        is_now_safe = not new.intersects(h_buffer)
        
        if was_unsafe and is_moved and is_now_safe: tp += 1
        elif was_unsafe and (not is_moved or not is_now_safe): fn += 1
        elif not was_unsafe and is_moved: fp += 1
        else: tn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "F1": f1, "Precision": precision, "Recall": recall}

def run_displacement(highway, roads, config):
    displaced = []
    safe_dist = (config['h_width']/2) + (config['r_width']/2) + config['clearance']
    h_buffer = highway.buffer(safe_dist, cap_style=2)
    
    for road in roads:
        if road.intersects(h_buffer):
            shift = safe_dist * 1.15
            s_left = road.parallel_offset(shift, 'left', join_style=2)
            s_right = road.parallel_offset(shift, 'right', join_style=2)
            
            if not s_left.is_empty and not s_left.intersects(h_buffer):
                displaced.append(s_left)
            elif not s_right.is_empty and not s_right.intersects(h_buffer):
                displaced.append(s_right)
            else:
                displaced.append(road)
        else:
            displaced.append(road)
    return displaced, safe_dist

def generate_ai_report(metrics, api_key):
    if not api_key: return "⚠️ API Key missing."
    
    # Model Fallback List
    models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    genai.configure(api_key=api_key)
    
    prompt = f"""
    Act as a GIS Engineer. Summarize this displacement analysis:
    - Conflicts Resolved: {metrics['TP']}
    - Unresolved: {metrics['FN']}
    - F1 Score: {metrics['F1']:.2f}
    Is this result acceptable for a safe navigation map?
    """
    
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            response = model.generate_content(prompt)
            return f"**Analysis by {m}:**\n\n{response.text}"
        except:
            continue
    return "❌ All models failed. Check quota."

def mermaid(code: str, height: int = 350):
    components.html(
        f"""
        <div class="mermaid">
            {code}
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
        </script>
        """,
        height=height,
    )

# --- 5. PAGE DRAWING ---

def draw_about_page():
    st.markdown("""
    <div style="background: linear-gradient(to right, #1F2937, #111827); padding: 30px; border-radius: 15px; border-left: 8px solid #00C9FF; margin-bottom: 25px;">
        <h1 style="margin:0; color: white; font-size: 40px;">🚀 Spatial Displacement Engine</h1>
        <p style="margin:10px 0 0 0; font-size: 18px; color: #A0AEC0;">
            An AI-powered cartographic generalization tool solving feature overlaps.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <span class="badge" style="background-color: #3776AB;">🐍 Python</span>
        <span class="badge" style="background-color: #FF4B4B;">⚡ Streamlit</span>
        <span class="badge" style="background-color: #E67E22;">📐 Shapely (Geometry)</span>
        <span class="badge" style="background-color: #2ECC71;">🌍 Folium (Maps)</span>
        <span class="badge" style="background-color: #9B59B6;">🤖 Google Gemini</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📌 The Challenge")
        st.info("**Feature Overlap:** When vector maps scale down, thick highways obscure parallel secondary roads.")
        
        st.subheader("💡 Our Solution")
        st.success("**Geometric Optimization:** We calculate collision buffers and mathematically displace lower-priority features.")
        
        st.markdown("### 🧮 Safety Buffer Formula")
        st.markdown("We define the 'Danger Zone' using rendering widths:")
        st.latex(r"Buffer = \frac{Width_{Highway}}{2} + \frac{Width_{Road}}{2} + Clearance")

    with col2:
        st.subheader("⚙️ Algorithm Logic Flow")
        mermaid_code = """
        graph LR
            A[Start] --> B{Check Overlap}
            B -- Yes --> C[Calculate Offset]
            B -- No --> D[Keep Original]
            C --> E[Try Left Shift]
            E -- Safe --> F[Success]
            E -- Conflict --> G[Try Right Shift]
            G -- Safe --> F
            G -- Conflict --> H[Manual Review]
            
            style A fill:#1F2937,stroke:#00C9FF,stroke-width:2px
            style B fill:#1F2937,stroke:#00C9FF,stroke-width:2px
            style C fill:#1F2937,stroke:#00C9FF,stroke-width:2px
            style D fill:#2ECC71,stroke:#000,stroke-width:2px
            style E fill:#1F2937,stroke:#00C9FF,stroke-width:2px
            style F fill:#2ECC71,stroke:#000,stroke-width:2px
            style G fill:#E67E22,stroke:#000,stroke-width:2px
            style H fill:#E74C3C,stroke:#000,stroke-width:2px
        """
        mermaid(mermaid_code)

def draw_app_page():
    st.title("🗺️ Axes Systems: Spatial Intelligence")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Simulation Controls")
        
        st.subheader("👁️ Visualization Mode")
        view_mode = st.radio("Select View:", ["Engineering Plane (White)", "Real-World Map"], index=0)
        
        st.markdown("---")
        h_width = st.slider("Highway Width", 1.0, 10.0, 5.0)
        r_width = st.slider("Road Width", 1.0, 8.0, 3.0)
        clearance = st.slider("Min Clearance", 0.5, 5.0, 2.0)
        
        st.markdown("---")
        st.subheader("📍 Location Calibration")
        target_lat = st.number_input("Latitude", value=50.874, format="%.4f")
        target_lon = st.number_input("Longitude", value=8.024, format="%.4f")
        
        st.markdown("---")
        input_mode = st.radio("Source:", ["Paste Text", "Upload File"])
        raw_wkt = ""
        if input_mode == "Paste Text":
            raw_wkt = st.text_area("Paste WKT Data:", height=150)
        else:
            uploaded = st.file_uploader("Upload .wkt", type=['wkt', 'txt'])
            if uploaded: raw_wkt = uploaded.read().decode("utf-8")
        
        use_swipe = st.checkbox("Enable Split-Screen Swipe", value=False)
        
        # RESET REPORT ON NEW RUN
        if st.button("🚀 Run Analysis", type="primary"):
            st.session_state["run_analysis"] = True
            st.session_state["ai_report_text"] = "" # Reset old report

    # --- MAIN CONTENT ---
    if st.session_state["run_analysis"] and raw_wkt:
        # Processing
        lines = parse_wkt_data(raw_wkt)
        if len(lines) < 2:
            st.error("Need at least 2 lines.")
            return

        highway = max(lines, key=lambda x: x.length)
        roads = [l for l in lines if l != highway]
        
        cfg = {'h_width': h_width, 'r_width': r_width, 'clearance': clearance}
        fixed_roads, safe_dist = run_displacement(highway, roads, cfg)
        metrics = calculate_metrics(highway, roads, fixed_roads, safe_dist)
        
        # Prepare Data for View
        all_geoms = [highway] + roads + fixed_roads
        vis_all = normalize_to_target(all_geoms, target_lat, target_lon)
        vis_h = vis_all[0]
        vis_r_orig = vis_all[1 : 1+len(roads)]
        vis_r_fixed = vis_all[1+len(roads) :]
        
        # 1. Metrics
        st.subheader("📊 Performance Analytics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 Score", f"{metrics['F1']:.2f}")
        c2.metric("Precision", f"{metrics['Precision']:.2f}")
        c3.metric("Recall", f"{metrics['Recall']:.2f}")
        c4.metric("Conflicts Resolved", metrics['TP'])
        
        chart_data = pd.DataFrame([
            {"Type": "Fixed (TP)", "Count": metrics['TP']},
            {"Type": "Missed (FN)", "Count": metrics['FN']},
            {"Type": "Clean (TN)", "Count": metrics['TN']}
        ])
        st.altair_chart(alt.Chart(chart_data).mark_bar().encode(
            x='Type', y='Count', color=alt.Color('Type', scale=alt.Scale(scheme='spectral'))
        ).properties(height=200), use_container_width=True)
        
        # 2. AI Report (PERSISTENT STATE FIXED)
        st.divider()
        st.subheader("🤖 Gemini Intelligence Report")
        
        # Try loading key from secrets
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            has_secret_key = True
        except:
            api_key = ""
            has_secret_key = False
            
        if not has_secret_key:
            api_key = st.text_input("Enter Gemini API Key:", type="password")
            
        if st.button("Generate AI Assessment"):
            if api_key:
                with st.spinner("Analyzing geometric patterns..."):
                    report = generate_ai_report(metrics, api_key)
                    # SAVE TO STATE
                    st.session_state["ai_report_text"] = report
            else:
                st.warning("Please add GEMINI_API_KEY to secrets.toml or paste it above.")
        
        # DISPLAY FROM STATE (This prevents disappearing)
        if st.session_state["ai_report_text"]:
            st.success(st.session_state["ai_report_text"])

        # 3. Interactive Map
        st.divider()
        st.subheader("📍 Interactive Viewer")
        
        tile_provider = "OpenStreetMap" if view_mode == "Real-World Map" else None
        attr_val = None if view_mode == "Real-World Map" else "Engineering Plane"

        # Bounds Calc
        all_lats, all_lons = [], []
        def get_coords(g_list):
            for g in g_list:
                if g.is_empty: continue
                parts = g.geoms if isinstance(g, MultiLineString) else [g]
                for p in parts:
                    for c in p.coords:
                        all_lons.append(c[0])
                        all_lats.append(c[1])
        get_coords([vis_h])
        get_coords(vis_r_orig)
        
        if not all_lats:
            st.error("No valid coordinates.")
            return

        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)

        m = folium.Map(
            location=[(min_lat+max_lat)/2, (min_lon+max_lon)/2], 
            zoom_start=14, 
            tiles=tile_provider, 
            attr=attr_val
        )
        
        fg_base = folium.FeatureGroup(name="Highway")
        add_to_map(vis_h, fg_base, "orange", h_width+2, "Highway")
        fg_base.add_to(m)
        
        fg_before = folium.FeatureGroup(name="Before (Red)")
        for r in vis_r_orig: add_to_map(r, fg_before, "red", r_width, "Original")
        fg_before.add_to(m)
        
        fg_after = folium.FeatureGroup(name="After (Green)")
        for r in vis_r_fixed: add_to_map(r, fg_after, "green", r_width, "Displaced")
        fg_after.add_to(m)
        
        if use_swipe:
            SideBySideLayers(layer_left=fg_before, layer_right=fg_after).add_to(m)
            
        folium.LayerControl().add_to(m)
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        
        st_folium(m, width="100%", height=600, key="main_map")

def main():
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio("Go to:", ["App Dashboard", "About Project"])
        st.markdown("---")
        
    if page == "App Dashboard":
        draw_app_page()
    else:
        draw_about_page()

if __name__ == "__main__":
    main()