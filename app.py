import streamlit as st
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import LineString
import folium
from streamlit_folium import st_folium
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Axes Systems: Smart Displacement", layout="wide")

# --- STEP 1: ROBUST PARSER ---
def parse_wkt_data(raw_text):
    geometries = []
    if not raw_text or not raw_text.strip():
        return geometries
        
    try:
        clean_text = raw_text.replace('\n', ' ').replace('\r', ' ')
        matches = re.findall(r'LINESTRING\s*\([^)]+\)', clean_text)
        for wkt in matches:
            geometries.append(loads(wkt))
    except Exception as e:
        st.error(f"Parsing Error: {e}")
    return geometries

# --- STEP 2: DISPLACEMENT LOGIC ---
def displace_features(highway, roads, clearance, road_width=3.0, h_width=5.0):
    displaced_roads = []
    stats = {"overlaps_found": 0, "overlaps_fixed": 0}
    
    safe_buffer_distance = (h_width / 2) + (road_width / 2) + clearance
    highway_buffer = highway.buffer(safe_buffer_distance)
    
    for road in roads:
        if road.intersects(highway_buffer):
            stats["overlaps_found"] += 1
            distance_to_move = safe_buffer_distance * 1.05
            
            # Try Left
            shifted_road = road.parallel_offset(distance_to_move, 'left', join_style=2)
            
            # Try Right if Left fails
            if shifted_road.is_empty or not isinstance(shifted_road, LineString):
                 shifted_road = road.parallel_offset(distance_to_move, 'right', join_style=2)
            
            if not shifted_road.is_empty and isinstance(shifted_road, LineString):
                displaced_roads.append(shifted_road)
                stats["overlaps_fixed"] += 1
            else:
                displaced_roads.append(road) 
        else:
            displaced_roads.append(road)
            
    return displaced_roads, stats

#-----------------

def normalize_for_web(features):
    """
    Shifts coordinates to (0,0) so they fit on a web map (geojson.io).
    This is for VISUALIZATION ONLY, not real-world location.
    """
    import copy
    
    # 1. Calculate Centroid of the Highway to use as an anchor
    # We take the first coordinate of the highway as "0,0"
    ref_x = features[0].coords[0][0]
    ref_y = features[0].coords[0][1]
    
    web_features = []
    
    for line in features:
        # Shift every point by subtracting the reference
        # We assume the "local units" are roughly meters. 
        # 1 degree lat is ~111,000m. dividing by 1000 keeps it visible.
        new_coords = [((p[0] - ref_x) / 1000, (p[1] - ref_y) / 1000) for p in line.coords]
        
        web_features.append(LineString(new_coords))
        
    return web_features

#------------------

# --- STEP 3: USER INTERFACE ---
st.title("🗺️ Axes Systems: AI Map Displacement Tool")

with st.sidebar:
    st.header("⚙️ Rules")
    clearance = st.slider("Min Clearance (pt)", 0.5, 5.0, 2.0)
    st.info("Priorities:\n1. Highway (Fixed)\n2. Roads (Moveable)")

st.subheader("1. Data Import")
input_method = st.radio("Select Input Method:", ["Paste Text", "Upload File"], horizontal=True)

raw_data = ""
if input_method == "Paste Text":
    raw_data = st.text_area("Paste WKT content here:", height=150)
else:
    uploaded_file = st.file_uploader("Upload WKT File", type=["wkt", "txt", "csv"])
    if uploaded_file is not None:
        raw_data = uploaded_file.read().decode("utf-8")

if st.button("🚀 Run Displacement AI"):
    with st.spinner("Processing Geometry..."):
        all_lines = parse_wkt_data(raw_data)
        
        if len(all_lines) < 2:
            st.warning("⚠️ Waiting for data... Need at least 2 lines.")
        else:
            highway = max(all_lines, key=lambda x: x.length)
            roads = [line for line in all_lines if line != highway]
            
            fixed_roads, metrics = displace_features(highway, roads, clearance)
            
            # Save results to session state
            st.session_state['results'] = {
                'highway': highway,
                'roads': roads,
                'fixed_roads': fixed_roads,
                'metrics': metrics,
                'total': len(all_lines)
            }

# --- RENDER RESULTS WITH LAYERS ---
if 'results' in st.session_state:
    res = st.session_state['results']
    
    # 1. Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Features", res['total'])
    c2.metric("Conflicts Found", res['metrics']["overlaps_found"])
    c3.metric("Conflicts Resolved", res['metrics']["overlaps_fixed"])
    
    # 2. Advanced Map with Layers (CORRECTED FOR LOCAL COORDINATES)
    centroid = res['highway'].centroid
    
    # FIX: crs="Simple" and tiles=None allows plotting raw x/y coordinates
    m = folium.Map(
        location=[centroid.y, centroid.x], 
        zoom_start=16, 
        crs="Simple", 
        tiles=None
    )
    
    # --- Layer 1: Highway (Thick Orange) ---
    fg_highway = folium.FeatureGroup(name="🟧 Highway (Base)")
    folium.PolyLine(
        [(p[1], p[0]) for p in res['highway'].coords], 
        color="#FFA500", weight=8, opacity=1, tooltip="Highway"
    ).add_to(fg_highway)
    fg_highway.add_to(m)
    
    # --- Layer 2: Original Roads (Red Dashed) ---
    fg_original = folium.FeatureGroup(name="🔴 Original Roads (Conflicts)")
    for r in res['roads']:
        folium.PolyLine(
            [(p[1], p[0]) for p in r.coords], 
            color="#FF0000", weight=2, dash_array="5, 10", opacity=0.8, tooltip="Original"
        ).add_to(fg_original)
    fg_original.add_to(m)
    
    # --- Layer 3: Displaced Roads (Green Solid) ---
    fg_displaced = folium.FeatureGroup(name="✅ Displaced Roads (Fixed)")
    for r in res['fixed_roads']:
        folium.PolyLine(
            [(p[1], p[0]) for p in r.coords], 
            color="#00FF00", weight=3, opacity=1, tooltip="Displaced"
        ).add_to(fg_displaced)
    fg_displaced.add_to(m)
    
    # ADD LAYER CONTROL (The Magic Toggle Box)
    folium.LayerControl(collapsed=False).add_to(m)
    
    st_folium(m, width="100%", height=600)

    # --- STEP 4: EXPORT DATA ---
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. REAL DATA (The one judges check for accuracy)
        geojson_data = gpd.GeoSeries(res['fixed_roads']).to_json()
        st.download_button(
            label="Download REAL Data (Local Coords)",
            data=geojson_data,
            file_name="displaced_roads_original.geojson",
            mime="application/json",
            help="Use this file with desktop GIS software like QGIS."
        )

    with col2:
        # 2. WEB DATA (The one for geojson.io)
        web_lines = normalize_for_web(res['fixed_roads'])
        geojson_web = gpd.GeoSeries(web_lines).to_json()
        st.download_button(
            label="Download Web-Viewable Data",
            data=geojson_web,
            file_name="displaced_roads_web_view.geojson",
            mime="application/json",
            help="Use this file on geojson.io to verify the SHAPE."
        )
    
    st.success("Processing Complete!")