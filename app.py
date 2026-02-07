import streamlit as st
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import LineString
import folium
from folium.plugins import SideBySideLayers
from streamlit_folium import st_folium
import re
import pandas as pd
import altair as alt
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Axes Systems: Spatial Intelligence", layout="wide", page_icon="🗺️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    div[data-testid="stMetric"] {
        background-color: #1F2937; padding: 15px; border-radius: 10px;
        border-left: 5px solid #00C9FF; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- GEMINI RAG CONFIGURATION ---
# INSTRUCTION: Get your key from https://aistudio.google.com/
# Set it in your secrets or paste it below (not recommended for production)
GEMINI_API_KEY = st.sidebar.text_input("🔑 Gemini API Key (for Explanations)", type="password")

def get_gemini_explanation(stats_summary, context_text):
    """
    RAG Function: Sends geometric data to Gemini for a natural language explanation.
    """
    if not GEMINI_API_KEY:
        return "⚠️ Please enter a Gemini API Key in the sidebar to generate an AI explanation."
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a senior Civil Engineer AI. Analyze this displacement report.
        
        DATA CONTEXT:
        We have a highway and local roads. We moved roads that were too close to the highway.
        
        STATISTICS:
        {stats_summary}
        
        TECHNICAL DETAILS:
        {context_text}
        
        TASK:
        1. Summarize what happened in plain English for a city planner.
        2. Explain the "Safety Score" (F1 Score) and what it implies about the algorithm's performance.
        3. Point out any specific roads that moved significantly.
        """
        
        with st.spinner("🤖 AI is analyzing the geometry..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"Error connecting to AI: {e}"

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

def calculate_advanced_metrics(highway, original_roads, displaced_roads, clearance, road_width=3.0, h_width=5.0):
    """
    Calculates Confusion Matrix elements and F1 Score for the geometric operation.
    """
    safe_dist = (h_width/2) + (road_width/2) + clearance
    highway_buffer = highway.buffer(safe_dist)
    
    tp = 0 # Needed moving & Moved successfully
    fp = 0 # Didn't need moving & Moved (Efficiency loss)
    fn = 0 # Needed moving & NOT Moved (Safety violation)
    tn = 0 # Didn't need moving & NOT Moved
    
    details = []
    
    for orig, new in zip(original_roads, displaced_roads):
        was_unsafe = orig.intersects(highway_buffer)
        is_moved = orig != new
        is_now_safe = not new.intersects(highway_buffer)
        
        if was_unsafe and is_moved and is_now_safe:
            tp += 1
            status = "TP (Fixed)"
        elif was_unsafe and (not is_moved or not is_now_safe):
            fn += 1
            status = "FN (Safety Fail)"
        elif not was_unsafe and is_moved:
            fp += 1
            status = "FP (Unnecessary Move)"
        else:
            tn += 1
            status = "TN (Clean)"
            
        # Calculate shift distance
        shift = 0
        if is_moved:
            try:
                shift = orig.centroid.distance(new.centroid)
            except: pass
            
        details.append({"Status": status, "Shift": shift})

    # F1 Score Calculation
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "scores": {"Precision": precision, "Recall": recall, "F1": f1},
        "details": details
    }

def displace_features(highway, roads, clearance):
    displaced_roads = []
    road_width = 3.0
    h_width = 5.0
    safe_buffer_distance = (h_width / 2) + (road_width / 2) + clearance
    highway_buffer = highway.buffer(safe_buffer_distance)
    
    for road in roads:
        if road.intersects(highway_buffer):
            distance_to_move = safe_buffer_distance * 1.05
            shifted = road.parallel_offset(distance_to_move, 'left', join_style=2)
            if shifted.is_empty or not isinstance(shifted, LineString):
                 shifted = road.parallel_offset(distance_to_move, 'right', join_style=2)
            
            if not shifted.is_empty and isinstance(shifted, LineString):
                displaced_roads.append(shifted)
            else:
                displaced_roads.append(road) # Fail safe
        else:
            displaced_roads.append(road)
            
    return displaced_roads

# --- MAIN APP ---
st.title("🗺️ Axes Systems: AI Geometric Optimization")

with st.sidebar:
    st.header("⚙️ Controls")
    clearance = st.slider("Clearance Buffer", 0.5, 5.0, 2.0)
    input_method = st.radio("Input:", ["Upload File", "Paste Text"])
    raw_data = ""
    if input_method == "Paste Text":
        raw_data = st.text_area("Paste WKT:", height=150)
    else:
        uploaded_file = st.file_uploader("Upload WKT", type=["wkt", "txt"])
        if uploaded_file is not None: raw_data = uploaded_file.read().decode("utf-8")

if st.button("🚀 Run Analysis"):
    all_lines = parse_wkt_data(raw_data)
    if len(all_lines) > 1:
        highway = max(all_lines, key=lambda x: x.length)
        roads = [line for line in all_lines if line != highway]
        
        # 1. Run Displacement
        fixed_roads = displace_features(highway, roads, clearance)
        
        # 2. Calculate Advanced Metrics
        metrics = calculate_advanced_metrics(highway, roads, fixed_roads, clearance)
        
        st.session_state['data'] = {
            'highway': highway, 'roads': roads, 'fixed': fixed_roads, 'metrics': metrics
        }
    else:
        st.error("Need at least 2 lines.")

if 'data' in st.session_state:
    d = st.session_state['data']
    m_res = d['metrics']
    
    # --- TABBED INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["📍 Visualizer", "📊 Analytics & F1", "🤖 AI Report (RAG)"])
    
    with tab1:
        st.subheader("Interactive Displacement Map")
        centroid = d['highway'].centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=16, crs="Simple", tiles=None)
        
        folium.PolyLine([(p[1], p[0]) for p in d['highway'].coords], color="orange", weight=8).add_to(m)
        
        fg_orig = folium.FeatureGroup(name="Before")
        for r in d['roads']: folium.PolyLine([(p[1], p[0]) for p in r.coords], color="red", weight=2).add_to(fg_orig)
        
        fg_new = folium.FeatureGroup(name="After")
        for r in d['fixed']: folium.PolyLine([(p[1], p[0]) for p in r.coords], color="#00FF00", weight=3).add_to(fg_new)
        
        SideBySideLayers(layer_left=fg_orig, layer_right=fg_new).add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, width="100%")

    with tab2:
        st.subheader("Performance Metrics")
        
        # F1 SCORE ROW
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1 Score", f"{m_res['scores']['F1']:.2f}")
        c2.metric("Precision", f"{m_res['scores']['Precision']:.2f}")
        c3.metric("Recall", f"{m_res['scores']['Recall']:.2f}")
        c4.metric("Total Roads", len(d['roads']))
        
        # CONFUSION MATRIX VISUALIZATION
        st.markdown("### Confusion Matrix (Geometric Accuracy)")
        matrix_data = pd.DataFrame([
            {"Actual": "Unsafe", "Predicted": "Moved", "Count": m_res['matrix']['TP'], "Type": "True Positive"},
            {"Actual": "Unsafe", "Predicted": "Static", "Count": m_res['matrix']['FN'], "Type": "False Negative"},
            {"Actual": "Safe", "Predicted": "Moved", "Count": m_res['matrix']['FP'], "Type": "False Positive"},
            {"Actual": "Safe", "Predicted": "Static", "Count": m_res['matrix']['TN'], "Type": "True Negative"}
        ])
        
        cm_chart = alt.Chart(matrix_data).mark_rect().encode(
            x='Predicted:N',
            y='Actual:N',
            color='Count:Q',
            tooltip=['Type', 'Count']
        ).properties(title="Confusion Matrix")
        
        text = cm_chart.mark_text(baseline='middle').encode(text='Count:Q', color=alt.value('white'))
        st.altair_chart(cm_chart + text, use_container_width=True)
        
        # DISPLACEMENT HISTOGRAM
        st.markdown("### Displacement Magnitude Distribution")
        shifts = [x['Shift'] for x in m_res['details'] if x['Shift'] > 0]
        if shifts:
            df_hist = pd.DataFrame({"Displacement (m)": shifts})
            hist = alt.Chart(df_hist).mark_bar(color='#00C9FF').encode(
                x=alt.X("Displacement (m)", bin=True),
                y='count()'
            )
            st.altair_chart(hist, use_container_width=True)

    with tab3:
        st.subheader("🤖 AI Executive Summary (RAG)")
        st.info("This module uses Gemini to explain the geometric changes.")
        
        if st.button("Generate AI Explanation"):
            # Prepare context for RAG
            stats_text = f"""
            Total Roads: {len(d['roads'])}
            Conflicts Detected (TP): {m_res['matrix']['TP']}
            Unresolved Conflicts (FN): {m_res['matrix']['FN']}
            F1 Score: {m_res['scores']['F1']:.2f}
            Avg Displacement: {sum(shifts)/len(shifts) if shifts else 0:.2f} meters.
            """
            
            explanation = get_gemini_explanation(stats_text, "Algorithm: Parallel Offset Displacement.")
            st.markdown(explanation)