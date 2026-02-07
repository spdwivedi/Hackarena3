import streamlit as st
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import LineString
from shapely.affinity import translate, scale
import folium
from folium.plugins import SideBySideLayers
from streamlit_folium import st_folium
import re
import pandas as pd
import altair as alt
import google.generativeai as genai

# --- CONFIGURATION & STYLE ---
st.set_page_config(page_title="Axes Systems: Spatial Intelligence", layout="wide", page_icon="🗺️")
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    div[data-testid="stMetric"] {
        background-color: #1F2937; border-radius: 10px;
        border-left: 5px solid #00C9FF; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- SECRETS MANAGEMENT (Auto-Load Key) ---
# Try to get key from Streamlit Secrets, otherwise warn user
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        GEMINI_API_KEY = None
except FileNotFoundError:
    GEMINI_API_KEY = None

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("🤖 AI Configuration")
    
    # Status Indicator
    if GEMINI_API_KEY:
        st.success("✅ API Key Loaded Securely")
    else:
        st.error("❌ No API Key Found in Secrets")
        st.info("Add your key to .streamlit/secrets.toml (Local) or Streamlit Cloud Settings.")

    # VALID Model Options (Only ones that exist today)
    model_options = {
        "Gemini 1.5 Pro (Best for Reasoning)": "gemini-1.5-pro",
        "Gemini 1.5 Flash (Fastest)": "gemini-1.5-flash",
        "Gemini 1.0 Pro (Standard)": "gemini-pro"
    }
    selected_model_name = st.selectbox("Select Model:", list(model_options.keys()))
    SELECTED_MODEL_ID = model_options[selected_model_name]

    st.markdown("---")
    st.header("⚙️ Simulation Controls")
    clearance = st.slider("Clearance Buffer", 0.5, 5.0, 2.0)
    input_method = st.radio("Input:", ["Upload File", "Paste Text"])
    
    raw_data = ""
    if input_method == "Paste Text": 
        raw_data = st.text_area("Paste WKT:")
    else:
        uploaded_file = st.file_uploader("Upload WKT", type=["wkt", "txt"])
        if uploaded_file is not None: 
            raw_data = uploaded_file.read().decode("utf-8")

# --- RAG FUNCTION ---
def get_gemini_explanation(stats_summary, context_text):
    if not GEMINI_API_KEY:
        return "⚠️ API Key missing. Please configure Streamlit Secrets."
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(SELECTED_MODEL_ID)
        
        prompt = f"""
        You are a Civil Engineer AI. Analyze this displacement report.
        STATS: {stats_summary}
        CONTEXT: {context_text}
        TASK: Summarize the safety improvements and explain the F1 score in simple terms.
        """
        with st.spinner(f"🤖 AI is analyzing..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e: 
        return f"Error: {e}"

# --- GEOMETRY & UTILS ---
def parse_wkt_data(raw_text):
    geometries = []
    if not raw_text or not raw_text.strip(): return geometries
    try:
        clean_text = raw_text.replace('\n', ' ').replace('\r', ' ')
        matches = re.findall(r'LINESTRING\s*\([^)]+\)', clean_text)
        for wkt in matches: geometries.append(loads(wkt))
    except Exception as e: st.error(f"Parsing Error: {e}")
    return geometries

def normalize_to_latlon_auto_scale(geometries):
    if not geometries: return []
    min_x = min(min(p[0] for p in g.coords) for g in geometries)
    min_y = min(min(p[1] for p in g.coords) for g in geometries)
    max_x = max(max(p[0] for p in g.coords) for g in geometries)
    max_y = max(max(p[1] for p in g.coords) for g in geometries)
    
    width = max_x - min_x if (max_x - min_x) > 0 else 1
    height = max_y - min_y if (max_y - min_y) > 0 else 1
    
    target_size = 0.02
    scale_factor = min(target_size/width, target_size/height)
    
    center_x, center_y = (min_x + max_x)/2, (min_y + max_y)/2
    
    vis_geoms = []
    for g in geometries:
        shifted = translate(g, xoff=-center_x, yoff=-center_y)
        scaled = scale(shifted, xfact=scale_factor, yfact=scale_factor, origin=(0,0))
        final = translate(scaled, xoff=-74.0060, yoff=40.7128)
        vis_geoms.append(final)
    return vis_geoms

def calculate_advanced_metrics(highway, original_roads, displaced_roads, clearance):
    road_width, h_width = 3.0, 5.0
    safe_dist = (h_width/2) + (road_width/2) + clearance
    highway_buffer = highway.buffer(safe_dist)
    
    tp, fp, fn, tn = 0, 0, 0, 0
    
    for orig, new in zip(original_roads, displaced_roads):
        was_unsafe = orig.intersects(highway_buffer)
        is_moved = orig != new
        is_now_safe = not new.intersects(highway_buffer)
        
        if was_unsafe and is_moved and is_now_safe: tp += 1
        elif was_unsafe and (not is_moved or not is_now_safe): fn += 1
        elif not was_unsafe and is_moved: fp += 1
        else: tn += 1
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}, "scores": {"F1": f1, "Precision": precision, "Recall": recall}}

def displace_features(highway, roads, clearance):
    displaced_roads = []
    safe_dist = (5.0/2) + (3.0/2) + clearance
    highway_buffer = highway.buffer(safe_dist)
    
    for road in roads:
        if road.intersects(highway_buffer):
            dist = safe_dist * 1.05
            shifted = road.parallel_offset(dist, 'left', join_style=2)
            if shifted.is_empty or not isinstance(shifted, LineString):
                 shifted = road.parallel_offset(dist, 'right', join_style=2)
            displaced_roads.append(shifted if not shifted.is_empty and isinstance(shifted, LineString) else road)
        else:
            displaced_roads.append(road)
    return displaced_roads

# --- MAIN APP LOGIC ---
if st.button("🚀 Run Analysis"):
    all_lines = parse_wkt_data(raw_data)
    if len(all_lines) > 1:
        highway = max(all_lines, key=lambda x: x.length)
        roads = [line for line in all_lines if line != highway]
        
        fixed_roads = displace_features(highway, roads, clearance)
        metrics = calculate_advanced_metrics(highway, roads, fixed_roads, clearance)
        
        # Visualize
        vis_all = normalize_to_latlon_auto_scale([highway] + roads + fixed_roads)
        st.session_state['data'] = {
            'vis_highway': vis_all[0],
            'vis_roads': vis_all[1:len(roads)+1],
            'vis_fixed': vis_all[len(roads)+1:],
            'metrics': metrics,
            'total': len(roads)
        }
    else: st.error("Need at least 2 lines.")

if 'data' in st.session_state:
    d = st.session_state['data']
    m_res = d['metrics']
    
    tab1, tab2, tab3 = st.tabs(["📍 Visualizer", "📊 Analytics", "🤖 AI Report"])
    
    with tab1:
        c = d['vis_highway'].centroid
        m = folium.Map(location=[c.y, c.x], zoom_start=15, tiles="CartoDB dark_matter")
        folium.PolyLine([(p[1], p[0]) for p in d['vis_highway'].coords], color="orange", weight=6).add_to(m)
        
        fg_orig = folium.FeatureGroup(name="Before")
        for r in d['vis_roads']: folium.PolyLine([(p[1], p[0]) for p in r.coords], color="red", weight=2).add_to(fg_orig)
        fg_orig.add_to(m)
        
        fg_new = folium.FeatureGroup(name="After")
        for r in d['vis_fixed']: folium.PolyLine([(p[1], p[0]) for p in r.coords], color="#00FF00", weight=2).add_to(fg_new)
        fg_new.add_to(m)
        
        SideBySideLayers(layer_left=fg_orig, layer_right=fg_new).add_to(m)
        folium.LayerControl().add_to(m)
        
        sw, ne = d['vis_highway'].bounds[0:2], d['vis_highway'].bounds[2:4]
        m.fit_bounds([[sw[1], sw[0]], [ne[1], ne[0]]])
        st_folium(m, width="100%", height=500)

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 Score", f"{m_res['scores']['F1']:.2f}")
        c2.metric("Precision", f"{m_res['scores']['Precision']:.2f}")
        c3.metric("Recall", f"{m_res['scores']['Recall']:.2f}")
        c4.metric("Total", d['total'])
        
        df = pd.DataFrame([{"Type": k, "Count": v} for k,v in m_res['matrix'].items()])
        st.altair_chart(alt.Chart(df).mark_bar().encode(x='Type', y='Count', color='Type'), use_container_width=True)

    with tab3:
        if st.button("Generate AI Explanation"):
            stats = f"TP: {m_res['matrix']['TP']}, FN: {m_res['matrix']['FN']}, F1: {m_res['scores']['F1']:.2f}"
            st.markdown(get_gemini_explanation(stats, "Geometric Displacement Algorithm"))