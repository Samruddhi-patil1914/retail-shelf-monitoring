"""
Retail Shelf Monitoring System - Streamlit Dashboard (FR-8, TRD Section 2)
Web dashboard for retail shelf image analysis, object detection, misplacement tracking,
empty-space detection, analytics visualization, and reporting.
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import time
from datetime import datetime
from pathlib import Path
import sys

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    LOW_STOCK_THRESHOLD_PCT,
    SAMPLES_DIR,
    UPLOADS_DIR,
    EXPORTS_DIR
)
from database.db_manager import DatabaseManager
from src.preprocessor import ImagePreprocessor
from src.detector import ShelfProductDetector
from src.shelf_analyzer import ShelfAnalyzer
from src.report_generator import ReportGenerator
from src.utils import draw_shelf_annotations

# Set page configuration
st.set_page_config(
    page_title="Retail Shelf Monitoring AI",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-good {
        color: #10B981;
        font-weight: bold;
    }
    .status-attention {
        color: #F59E0B;
        font-weight: bold;
    }
    .status-critical {
        color: #EF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db():
    return DatabaseManager()

@st.cache_resource
def get_detector():
    return ShelfProductDetector()

db = get_db()
detector = get_detector()

# ================= SIDEBAR =================
st.sidebar.image("https://img.icons8.com/fluency/96/supermarket-shelf.png", width=70)
st.sidebar.title("Shelf AI Control")

st.sidebar.markdown("### ?? Detection Settings")
conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=float(DEFAULT_CONFIDENCE_THRESHOLD),
    step=0.05,
    help="Minimum confidence score for YOLO to register a detection."
)

low_stock_thresh = st.sidebar.slider(
    "Low Stock Threshold (%)",
    min_value=10,
    max_value=60,
    value=int(LOW_STOCK_THRESHOLD_PCT),
    step=5,
    help="Shelf occupancy percentage below which an alert is triggered."
)

show_zones = st.sidebar.checkbox("Overlay Shelf Zones", value=True)
show_gaps = st.sidebar.checkbox("Highlight Empty Space Gaps", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ?? Quick Demo Presets")
sample_choice = st.sidebar.selectbox(
    "Load Pre-packaged Sample:",
    ["(Upload Custom Image)", "Normal Stocked Shelf", "Low Stock & Empty Gaps", "Misplaced Products Shelf"]
)

st.sidebar.markdown("---")
model_status = "? Ultralytics YOLOv8n" if detector.is_real_yolo else "?? Heuristic CV Segmenter"
st.sidebar.info(f"**Active Inference Engine:** {model_status}")

# ================= MAIN PAGE HEADER =================
st.markdown('<div class="main-header">?? Retail Shelf Monitoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Product Detection, Stock Level Auditing & Planogram Compliance using Computer Vision</div>', unsafe_allow_html=True)

# Image input handling
uploaded_file = st.file_uploader(
    "Upload Shelf Photo (JPG, PNG)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an image of a retail display rack to run inventory and planogram inspection."
)

active_image_source = None
image_name = "custom_upload"

if uploaded_file is not None:
    active_image_source = uploaded_file.getvalue()
    image_name = uploaded_file.name
elif sample_choice == "Normal Stocked Shelf":
    sample_path = SAMPLES_DIR / "shelf_normal.png"
    if sample_path.exists():
        active_image_source = str(sample_path)
        image_name = "shelf_normal.png"
elif sample_choice == "Low Stock & Empty Gaps":
    sample_path = SAMPLES_DIR / "shelf_low_stock.png"
    if sample_path.exists():
        active_image_source = str(sample_path)
        image_name = "shelf_low_stock.png"
elif sample_choice == "Misplaced Products Shelf":
    sample_path = SAMPLES_DIR / "shelf_misplaced.png"
    if sample_path.exists():
        active_image_source = str(sample_path)
        image_name = "shelf_misplaced.png"

# Default fallback if nothing selected
if active_image_source is None:
    sample_path = SAMPLES_DIR / "shelf_normal.png"
    if sample_path.exists():
        active_image_source = str(sample_path)
        image_name = "shelf_normal.png"

# Processing pipeline
if active_image_source is not None:
    start_time = time.time()

    # Load & Preprocess
    original_rgb, model_input, scale, pad = ImagePreprocessor.preprocess_pipeline(active_image_source)

    # Detect
    detections = detector.detect(original_rgb, conf_threshold=conf_thresh)

    # Analyze
    zones = db.get_all_zones()
    analyzer = ShelfAnalyzer(zones)
    results = analyzer.analyze(original_rgb, detections, low_stock_threshold=low_stock_thresh)

    # Annotate
    annotated_rgb = draw_shelf_annotations(
        original_rgb,
        results['detections'],
        zones,
        empty_slots=results['empty_slots'],
        show_zones=show_zones,
        show_gaps=show_gaps
    )

    inference_duration = time.time() - start_time

    # Save to SQLite Database
    scan_id = f"SCAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.log_detections(scan_id, results['detections'])
    db.log_report(
        image_id=image_name,
        total_products=results['total_products'],
        empty_zones_count=results['empty_zones_count'],
        misplaced_count=results['misplaced_count'],
        shelf_status=results['shelf_status'],
        health_score=results['health_score']
    )

    # ================= KPI METRIC CARDS =================
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Products",
            value=results['total_products'],
            delta=f"{results['total_products']} items detected"
        )

    with col2:
        st.metric(
            label="Misplaced Items",
            value=results['misplaced_count'],
            delta="- Violations" if results['misplaced_count'] > 0 else "Compliant",
            delta_color="inverse" if results['misplaced_count'] > 0 else "normal"
        )

    with col3:
        st.metric(
            label="Empty Shelf Zones",
            value=results['empty_zones_count'],
            delta="- Attention" if results['empty_zones_count'] > 0 else "All Stocked",
            delta_color="inverse" if results['empty_zones_count'] > 0 else "normal"
        )

    with col4:
        st.metric(
            label="Low Stock Zones",
            value=results['low_stock_zones_count'],
            delta="Restock Soon" if results['low_stock_zones_count'] > 0 else "Good",
            delta_color="off"
        )

    with col5:
        score_val = results['health_score']
        status_label = results['shelf_status']
        st.metric(
            label="Shelf Health Score",
            value=f"{score_val}%",
            delta=status_label,
            delta_color="normal" if status_label == "Good" else "inverse"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= DETAIL TABS =================
    tab_view, tab_charts, tab_alerts, tab_summary, tab_data, tab_history = st.tabs([
        "??? Visual Inspection",
        "?? Stock Analytics",
        "?? Operational Alerts",
        "?? Audit Summary",
        "?? Export Data",
        "?? Scan History"
    ])

    # Tab 1: Visual Inspection
    with tab_view:
        st.subheader("Shelf Computer Vision Overlay")
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown("**Original Shelf Image**")
            st.image(original_rgb, use_container_width=True)
        with vcol2:
            st.markdown("**AI Detection & Zone Compliance**")
            st.image(annotated_rgb, use_container_width=True)

        st.caption(f"Inference latency: {inference_duration:.2f}s | Resolution: {original_rgb.shape[1]}x{original_rgb.shape[0]}")

    # Tab 2: Analytics & Charts
    with tab_charts:
        st.subheader("Category & Zone Inventory Distribution")
        ccol1, ccol2 = st.columns(2)

        with ccol1:
            st.markdown("#### Products by Category")
            if results['counts_by_category']:
                cat_df = pd.DataFrame(
                    list(results['counts_by_category'].items()),
                    columns=['Category', 'Count']
                )
                st.bar_chart(cat_df.set_index('Category'))
            else:
                st.info("No items detected to categorize.")

        with ccol2:
            st.markdown("#### Shelf Capacity by Zone")
            za_data = []
            for z in results['zone_analytics']:
                za_data.append({
                    'Zone': z['zone_name'],
                    'Stock %': z['stock_percentage'],
                    'Empty %': z['empty_percentage']
                })
            if za_data:
                za_df = pd.DataFrame(za_data).set_index('Zone')
                st.bar_chart(za_df)

        st.markdown("#### Zone Stock Status Breakdown")
        for z in results['zone_analytics']:
            stat_color = "green" if z['status'] == "Well Stocked" else ("orange" if z['status'] == "Low Stock" else "red")
            st.write(f"**{z['zone_name']}** (Target: `{z['expected_category']}`): {z['current_count']}/{z['capacity']} units ({z['stock_percentage']}%) ? :{stat_color}[{z['status']}]")
            st.progress(min(1.0, z['stock_percentage'] / 100.0))

    # Tab 3: Alerts
    with tab_alerts:
        st.subheader("Actionable Alerts for Store Staff")
        for alert in results['alerts']:
            sev = alert['severity']
            if sev == "CRITICAL":
                st.error(f"**{alert['title']}**: {alert['message']}")
            elif sev == "WARNING":
                st.warning(f"**{alert['title']}**: {alert['message']}")
            else:
                st.success(f"**{alert['title']}**: {alert['message']}")

    # Tab 4: Audit Summary
    with tab_summary:
        st.subheader("Executive Shelf Health Report")
        nl_text = ReportGenerator.generate_nl_summary(results)
        st.markdown(nl_text)

        st.markdown("---")
        st.markdown("#### ?? Optional Local LLM Summary (TRD FR-10)")
        if st.button("Generate with Ollama (Llama 3.2 3B)"):
            with st.spinner("Connecting to local Ollama runtime..."):
                llm_output = ReportGenerator.generate_nl_summary(results, use_ollama=True)
                st.info(llm_output)

    # Tab 5: Export Data
    with tab_data:
        st.subheader("Export Scan Data & Annotated Photos")
        df_export = ReportGenerator.generate_detections_dataframe(image_name, results)
        st.dataframe(df_export, use_container_width=True)

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            st.download_button(
                label="?? Download Detections CSV",
                data=csv_buffer.getvalue(),
                file_name=f"shelf_audit_{image_name}.csv",
                mime="text/csv"
            )

        with dcol2:
            img_pil = Image.fromarray(annotated_rgb)
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='PNG')
            st.download_button(
                label="??? Download Annotated Image",
                data=img_byte_arr.getvalue(),
                file_name=f"annotated_{image_name}.png",
                mime="image/png"
            )

    # Tab 6: Scan History
    with tab_history:
        st.subheader("Previous Shelf Scans from SQLite Database")
        history_df = db.get_recent_reports(limit=15)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No prior scan records in the database.")

else:
    st.info("Please upload an image or choose a demo sample from the sidebar to begin analysis.")
