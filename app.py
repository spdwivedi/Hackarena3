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

# --- SIDEBAR: MODEL & KEY CONFIGURATION ---
with st.sidebar:
    st.header("🤖 AI Configuration")
    GEMINI_API_KEY = st.text_input("🔑 Gemini API Key", type="password")
    
    # NEW: Model Selector with the versions you provided
    model_options = {
        "Gemini 3 Pro (Most Intelligent)": "gemini-3.0-pro-001",
        "Gemini 3 Flash (Balanced)": "gemini-3.0-flash-001", 
        "Gemini 2.5 Pro (Advanced Thinking)": "gemini-2.5-pro-latest",
        "Gemini 2.5 Flash (Fast & Intelligent)": "gemini-2.5-flash-latest",
        "Gemini 2.5 Flash-Lite (Ultra Fast)": "gemini-2.5-flash-lite-latest",
        "Gemini 1.5 Pro (Legacy Stable)": "gemini-1.5-pro",
        "Gemini 1.5 Flash (Legacy Fast)": "gemini-1.5-flash"
    }
    
    selected_model_name = st.selectbox("Select Model Version:", list(model_options.keys()))
    SELECTED_MODEL_ID = model_options[selected_model_name]

# --- RAG FUNCTION ---
def get_gemini_explanation(stats_summary, context_text):
    if not GEMINI_API_KEY:
        return "⚠️ Please enter a Gemini API Key in the sidebar to generate an AI explanation."
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Use the user-selected model
        model = genai.GenerativeModel(SELECTED_MODEL_ID)
        
        prompt = f"""
        You are a Civil Engineer AI using {selected_model_name}. Analyze this displacement report.
        STATS: {stats_summary}
        CONTEXT: {context_text}
        TASK: Summarize the safety improvements and explain the F1 score in simple terms.
        """
        with st.spinner(f"🤖 {selected_model_name} is analyzing..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e: 
        return f"Error connecting to AI: {e}. (Note: Verify if your API key has access to {SELECTED_MODEL_ID})"

# --- GEOMETRY UTILS ---
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
    """
    Robust Normalization: Fits ANY coordinate system into a viewable NYC block.
    """
    if not geometries: return []
    
    # 1. Get bounds of original data
    min_x = min(min(p[0] for p in g.coords) for g in geometries)
    min_y = min(min(p[1] for p in g.coords) for g in geometries)
    max_x = max(max(p[0] for p in g.coords) for g in geometries)
    max_y = max(max(p[1] for p in g.coords) for g in geometries)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Avoid division by zero
    if width == 0: width = 1
    if height == 0: height = 1
    
    # 2. Scale factor: We want the max dimension to be roughly 0.02 degrees (visible city view)
    target_size = 0.02
    scale_x = target_size / width
    scale_y = target_size / height
    final_scale = min(scale_x, scale_y) # Maintain aspect ratio
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    vis_geoms = []
    for g in geometries:
        # A. Center at (0,0)
        shifted = translate(g, xoff=-center_x, yoff=-center_y)
        # B. Scale to Lat/Lon size
        scaled = scale(shifted, xfact=final_scale, yfact=final_scale, origin=(0,0))
        # C. Move to NYC (Manhattan)
        final = translate(scaled, xoff=-74.0060, yoff=40.7128)
        vis_geoms.append(final)
        
    return vis_geoms

def calculate_advanced_metrics(highway, original_roads, displaced_roads, clearance):
    road_width, h_width = 3.0, 5.0
    safe_dist = (h_width/2) + (road_width/2) + clearance
    highway_buffer = highway.buffer(safe_dist)
    
    tp, fp, fn, tn = 0, 0, 0, 0
    details = []
    
    for orig, new in zip(original_roads, displaced_roads):
        was_unsafe = orig.intersects(highway_buffer)
        is_moved = orig != new
        is_now_safe = not new.intersects(highway_buffer)
        
        if was_unsafe and is_moved and is_now_safe: tp += 1; status="TP"
        elif was_unsafe and (not is_moved or not is_now_safe): fn += 1; status="FN"
        elif not was_unsafe and is_moved: fp += 1; status="FP"
        else: tn += 1; status="TN"
        
        shift = orig.centroid.distance(new.centroid) if is_moved else 0
        details.append({"Status": status, "Shift": shift})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}, "scores": {"F1": f1, "Precision": precision, "Recall": recall}, "details": details}

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

# --- MAIN APP ---
st.title("🗺️ Axes Systems: AI Geometric Optimization")

with st.sidebar:
    st.markdown("---")
    st.header("⚙️ Simulation Controls")
    clearance = st.slider("Clearance Buffer", 0.5, 5.0, 2.0)
    input_method = st.radio("Input:", ["Upload File", "Paste Text"])
    raw_data = ""
    if input_method == "Paste Text": raw_data = st.text_area("Paste WKT:")
    else:
        uploaded_file = st.file_uploader("Upload WKT", type=["wkt", "txt"])
        if uploaded_file is not None: raw_data = uploaded_file.read().decode("utf-8")

if st.button("🚀 Run Analysis"):
    all_lines = parse_wkt_data(raw_data)
    if len(all_lines) > 1:
        highway = max(all_lines, key=lambda x: x.length)
        roads = [line for line in all_lines if line != highway]
        
        # 1. Calculation
        fixed_roads = displace_features(highway, roads, clearance)
        metrics = calculate_advanced_metrics(highway, roads, fixed_roads, clearance)
        
        # 2. Visualization (Auto-Scaled to NYC)
        all_geoms = [highway] + roads + fixed_roads
        vis_all = normalize_to_latlon_auto_scale(all_geoms)
        
        vis_highway = vis_all[0]
        vis_roads = vis_all[1:len(roads)+1]
        vis_fixed = vis_all[len(roads)+1:]
        
        st.session_state['data'] = {
            'vis_highway': vis_highway, 'vis_roads': vis_roads, 'vis_fixed': vis_fixed,
            'metrics': metrics, 'total_roads': len(roads)
        }
    else: st.error("Need at least 2 lines.")

if 'data' in st.session_state:
    d = st.session_state['data']
    m_res = d['metrics']
    
    tab1, tab2, tab3 = st.tabs(["📍 Visualizer", "📊 Analytics", "🤖 AI Report"])
    
    with tab1:
        st.subheader("Interactive Map (Projected View)")
        
        # Determine map center dynamically
        if d['vis_highway']:
            c = d['vis_highway'].centroid
            start_loc = [c.y, c.x]
        else:
            start_loc = [40.7128, -74.0060]

        m = folium.Map(location=start_loc, zoom_start=15, tiles="CartoDB dark_matter")
        
        # Add Highway
        folium.PolyLine([(p[1], p[0]) for p in d['vis_highway'].coords], color="#FFA500", weight=6, opacity=1, tooltip="Highway").add_to(m)
        
        # Layer: Before
        fg_orig = folium.FeatureGroup(name="Before Displacement")
        for r in d['vis_roads']: 
            folium.PolyLine([(p[1], p[0]) for p in r.coords], color="#FF4B4B", weight=2, opacity=0.7).add_to(fg_orig)
        fg_orig.add_to(m) # Add to map so it's visible by default
            
        # Layer: After
        fg_new = folium.FeatureGroup(name="After Displacement")
        for r in d['vis_fixed']: 
            folium.PolyLine([(p[1], p[0]) for p in r.coords], color="#00FF00", weight=2, opacity=1).add_to(fg_new)
        fg_new.add_to(m) # Add to map so it's visible by default

        # Add Split Screen Slider
        SideBySideLayers(layer_left=fg_orig, layer_right=fg_new).add_to(m)
        folium.LayerControl().add_to(m)
        
        # IMPORTANT: Fit Bounds to ensure visibility
        sw = d['vis_highway'].bounds[0:2]
        ne = d['vis_highway'].bounds[2:4]
        # Folium expects [lat, lon], shapely gives [lon, lat]
        m.fit_bounds([[sw[1], sw[0]], [ne[1], ne[0]]])
        
        st_folium(m, width="100%", height=500)

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 Score", f"{m_res['scores']['F1']:.2f}")
        c2.metric("Precision", f"{m_res['scores']['Precision']:.2f}")
        c3.metric("Recall", f"{m_res['scores']['Recall']:.2f}")
        c4.metric("Roads Analyzed", d['total_roads'])
        
        matrix_df = pd.DataFrame([
            {"Type": "Fixed (TP)", "Count": m_res['matrix']['TP']},
            {"Type": "Missed (FN)", "Count": m_res['matrix']['FN']},
            {"Type": "Unnecessary (FP)", "Count": m_res['matrix']['FP']},
            {"Type": "Clean (TN)", "Count": m_res['matrix']['TN']}
        ])
        chart = alt.Chart(matrix_df).mark_bar().encode(x='Type', y='Count', color=alt.Color('Type', scale=alt.Scale(scheme='spectral')))
        st.altair_chart(chart, use_container_width=True)

    with tab3:
        if st.button("Generate AI Explanation"):
            stats = f"TP: {m_res['matrix']['TP']}, FN: {m_res['matrix']['FN']}, F1: {m_res['scores']['F1']:.2f}"
            st.markdown(get_gemini_explanation(stats, "Geometric Displacement Algorithm"))