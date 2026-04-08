import streamlit as st
import os
import sqlite3
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Product Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATABASE SETUP ---
DB_NAME = "defects.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS defects
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_name TEXT,
                  defect_type TEXT,
                  confidence REAL)''')
    conn.commit()
    conn.close()

def save_detection(image_name, defect_type, confidence):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO defects (image_name, defect_type, confidence) VALUES (?, ?, ?)",
              (image_name, defect_type, confidence))
    conn.commit()
    conn.close()

def get_all_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM defects", conn)
    conn.close()
    return df

init_db()

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Use the model path from the original project
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        # Fallback to yolov8n.pt if best.pt is missing
        model_path = "yolov8n.pt"
    return YOLO(model_path)

model = load_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Detection Home", "Analytics Dashboard"])

if page == "Detection Home":
    st.title("🔍 Automated Product Defect Detection")
    st.write("Upload an image to detect product defects using AI.")

    uploaded_file = st.file_uploader("Choose a product image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Detection Result")
            if st.button("Run Detection"):
                # Save temp file for YOLO
                temp_path = "temp_upload.jpg"
                image.save(temp_path)
                
                # Run YOLO
                results = model(temp_path, conf=0.05)
                res = results[0]
                
                # Plot results
                annotated_img = res.plot()
                # res.plot() returns BGR numpy array if using cv2, or RGB if specified. 
                # Ultralytics usually returns RGB for plot() in recent versions when using Image.fromarray later.
                st.image(annotated_img, channels="BGR", use_column_width=True)
                
                # Process boxes
                boxes = getattr(res, "boxes", [])
                if boxes is None or len(boxes) == 0:
                    st.info("No defects detected.")
                    save_detection(uploaded_file.name, "no_defect", 0.0)
                else:
                    st.success(f"Detected {len(boxes)} issue(s)!")
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = model.names.get(cls, str(cls))
                        save_detection(uploaded_file.name, name, round(conf * 100, 2))
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)

elif page == "Analytics Dashboard":
    st.title("📊 Defect Analytics Dashboard")
    
    df = get_all_data()
    
    if df.empty:
        st.warning("No data available yet. Please run some detections first!")
    else:
        # Metrics
        total_images = df['image_name'].nunique()
        defect_df = df[df['defect_type'] != 'no_defect']
        total_defects = len(defect_df)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Images Processed", total_images)
        m2.metric("Total Defects Found", total_defects)
        
        if not defect_df.empty:
            avg_conf = defect_df['confidence'].mean()
            m3.metric("Avg Detection Conf.", f"{avg_conf:.1f}%")
            
            # Charts
            st.divider()
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Defect Types Distribution")
                type_counts = defect_df['defect_type'].value_counts()
                st.bar_chart(type_counts)
                
            with c2:
                st.subheader("Defect Confidence (Last 20)")
                st.line_chart(defect_df.tail(20).set_index('id')['confidence'])
            
            st.divider()
            st.subheader("Recent Detection History")
            st.dataframe(df.sort_values(by='id', ascending=False), use_container_width=True)
        else:
            m3.metric("Avg Detection Conf.", "0%")
            st.write("Results will appear here once defects are detected.")

st.sidebar.divider()
st.sidebar.info("System Ready | YOLOv8 Powered")
