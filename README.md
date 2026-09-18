# Retail Shelf Monitoring Using Computer Vision and Data Science

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-YOLOv8-red?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Framework](https://img.shields.io/badge/Framework-Streamlit-orange?style=for-the-badge&logo=streamlit&logoColor=white)
![Database](https://img.shields.io/badge/Database-SQLite-yellow?style=for-the-badge&logo=sqlite&logoColor=black)
![Data](https://img.shields.io/badge/Data-Pandas-blue?style=for-the-badge&logo=pandas&logoColor=white)
![Numerical](https://img.shields.io/badge/Numerical-NumPy-purple?style=for-the-badge&logo=numpy&logoColor=white)

> **B.Sc. Data Science — Third Year Project**

An AI-powered computer vision system for retail shelf monitoring that detects products, analyzes shelf availability, identifies misplaced items, and provides actionable insights through an interactive Streamlit dashboard.

---
# 📌 Executive Summary & Problem Statement

Retail stores need efficient ways to monitor shelf availability, product placement, and stock levels. This project uses computer vision to analyze shelf images and provide useful retail insights through an interactive Streamlit dashb#oard.

---

## Project Objectives
1. **Product Detection & Localization**: Detect retail merchandise with bounding boxes using YOLOv8n (nano variant).
2. **Product Counting**: Automatically count inventory by category (bottles, cans, boxes, etc.) and shelf zone.
3. **Empty / Low-Stock Area Detection**: Identify vacant shelf segments, calculate percentage occupancy, and flag low-stock thresholds.
4. **Misplacement Detection (Planogram Compliance)**: Compare detected items against assigned shelf zones and flag misplaced products visually.
5. **Shelf Health Scoring**: Compute composite 0?100% health scores and assign statuses (*Good*, *Needs Attention*, *Critical*).
6. **Interactive Dashboard & Reporting**: Visualize side-by-side computer vision overlays, telemetry charts, and provide downloadable CSV audits and annotated images.

---

## ✨ Key Features

- 📷 **Shelf Image Analysis** — Upload retail shelf images for AI-based analysis.
- 🔍 **Product Detection** — Detect products using the YOLOv8 object detection model.
- 📦 **Product Counting** — Count detected products on the shelf.
- 🏪 **Shelf Availability Monitoring** — Identify empty and low-stock shelf zones.
- ⚠️ **Misplaced Product Detection** — Identify products that appear outside their expected shelf zones.
- 📊 **Interactive Dashboard** — View detection and shelf analytics through a Streamlit dashboard.
- 🗄️ **Scan History** — Store and review previous shelf analysis results.
- 📄 **Data Export** — Export analysis results for further use.
- 📈 **Shelf Health Analysis** — Generate an overall shelf condition score based on detected shelf conditions.

---

## 🛠️ Technologies Used
| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming and data processing |
| 🤖 YOLOv8 | Product object detection |
| 👁️ OpenCV | Image processing and computer vision |
| 📊 Pandas | Data processing and analysis |
| 📈 Matplotlib | Data visualization |
| 🌐 Streamlit | Interactive web dashboard |
| 🗄️ SQLite | Database and scan history storage |
| 🖼️ PIL | Image handling and processing |
| 🔧 Git & GitHub | Version control and project hosting |

---

## System Architecture & Workflow

```mermaid
## System Architecture & Workflow

flowchart TD
    A[Image Input<br>JPG / PNG] --> B[Preprocessing<br>OpenCV / PIL<br>Resize & Normalize]
    B --> C[YOLOv8 Object Detection<br>Products & Bounding Boxes]
    C --> D[Shelf Analysis]

    D --> D1[Product Counting]
    D --> D2[Empty-Space Detection]
    D --> D3[Misplacement Detection]
    D --> D4[Occupancy Analysis]

    D1 --> E[SQLite Database]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F[Streamlit Web Dashboard]
    F --> G[Charts & Alerts]
    F --> H[Reports & Export]
---

## Project Directory Structure

```text
retail_shelf_monitoring/
?
??? app.py                      # Main Streamlit web dashboard application
??? config.py                   # Global system configuration, paths, and thresholds
??? requirements.txt            # Python dependencies
??? README.md                   # Complete academic and operational documentation
?
??? src/                        # Core Application Layer
?   ??? __init__.py             # Package exports
?   ??? preprocessor.py         # Image validation, letterbox resizing, contrast enhancement (FR-2)
?   ??? detector.py             # Ultralytics YOLOv8n detector with resilient fallback (FR-3)
?   ??? shelf_analyzer.py       # Counting, gap analysis, misplacement check, health score (FR-4..7)
?   ??? report_generator.py     # CSV report builder and natural-language summaries (FR-9, FR-10)
?   ??? utils.py                # Color-coded CV bounding boxes, zones, and empty gap annotations
?
??? database/                   # Data Layer
?   ??? __init__.py
?   ??? db_manager.py           # SQLite manager for detections, zones, and reports (TRD Sec 9)
?
??? scripts/                    # Utilities & Machine Learning Pipeline
?   ??? generate_sample_data.py # Generates synthetic shelf images (Normal, Low Stock, Misplaced)
?   ??? train.py                # Transfer learning training script for custom shelf datasets (TRD Sec 7)
?
??? data/                       # Storage Layer
?   ??? uploads/                # Directory for user uploaded raw shelf photos
?   ??? exports/                # Exported CSV audit reports and annotated images
?   ??? samples/                # Pre-generated sample images for instant demonstration
?   ??? shelf_monitoring.db     # SQLite relational database
?
??? tests/                      # Verification Layer
    ??? __init__.py
    ??? run_tests.py            # Zero-dependency test runner using Python standard library
    ??? test_pipeline.py        # Pytest test suite for end-to-end pipeline verification
```

---

## File-by-File Breakdown

| File Path | Description & Responsibility |
| :--- | :--- |
| `config.py` | Central configuration containing bounding thresholds, default 3-tier shelf coordinates (`ZONE_TOP`, `ZONE_MID`, `ZONE_BOT`), category synonyms, and file paths. |
| `database/db_manager.py` | Encapsulates SQLite schema. Implements tables: `zones`, `detections`, and `reports`. Handles logging and historical queries. |
| `src/preprocessor.py` | Validates file formats and file size limits (?10MB). Performs letterbox aspect-ratio preserving resizing to 640x640 and CLAHE adaptive histogram equalization. |
| `src/detector.py` | Loads YOLOv8n (nano). Runs inference, filters bounding boxes by confidence threshold, and provides a robust feature-segmenter fallback for environments without CUDA/weights. |
| `src/shelf_analyzer.py` | Computes zone assignments, identifies misplaced items, tallies category counts, performs horizontal empty gap analysis, calculates stock percentage, and computes the 0?100% Shelf Health Score. |
| `src/utils.py` | Overlays color-coded bounding boxes (Emerald Green for compliant, Crimson Red for misplaced), dotted Amber boxes for empty shelf slots, and zone separator banners. |
| `src/report_generator.py` | Generates standardized CSV audit logs (`image_id, class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, zone, status`), structured markdown summaries, and optional Ollama LLM summaries. |
| `app.py` | Interactive Streamlit user interface featuring KPI metric cards, side-by-side CV visual inspection, Plotly bar/progress charts, operational alerts, and CSV/image export downloads. |
| `scripts/generate_sample_data.py` | Creates 3 realistic supermarket shelf test images (`shelf_normal.png`, `shelf_low_stock.png`, `shelf_misplaced.png`) and initializes the SQLite database. |
| `scripts/train.py` | Fine-tuning pipeline using PyTorch and Ultralytics YOLOv8n with configurable hyperparameters (epochs, batch size, learning rate, and dataset splits). |
| `tests/run_tests.py` | Automated test suite validating database initialization, preprocessing, object detection, misplacement detection, and report generation. |

---

## Installation & Setup Guide

### 1. Prerequisites
- **Operating System**: Windows 10/11, macOS, or Ubuntu Linux
- **Python**: Version 3.10 or higher installed

### 2. Environment Setup
Open PowerShell or your terminal in the project directory:

```bash
# Navigate to the project directory
cd retail_shelf_monitoring

# Create a clean virtual environment
python -m venv .venv

# Activate the virtual environment:
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (CMD):
.\.venv\Scriptsctivate.bat
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### Step 1: Generate Sample Images & Initialize Database
Run the sample data generator script to populate initial demonstration scenarios:
```bash
python scripts/generate_sample_data.py
```
*Output: Generates `shelf_normal.png`, `shelf_low_stock.png`, `shelf_misplaced.png` inside `data/samples/` and initializes `data/shelf_monitoring.db`.*

### Step 2: Run Unit Tests
Verify all modules and algorithms pass verification:
```bash
python tests/run_tests.py
```

### Step 3: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your web browser and navigate to:
`http://localhost:8501`

---

## 📊 Dashboard & Results

The Retail Shelf Monitoring system provides an interactive Streamlit dashboard for analyzing shelf images and viewing AI-generated insights.

### Dashboard Modules

- 🔍 **Visual Inspection** — View the uploaded shelf image and AI detection results.
- 📦 **Stock Analytics** — Analyze product counts and shelf availability.
- ⚠️ **Operational Alerts** — Identify low-stock, empty, and potentially misplaced shelf conditions.
- 📋 **Audit Summary** — Review the overall shelf analysis.
- 📥 **Export Data** — Export analysis results for further use.
- 🕒 **Scan History** — View previous shelf monitoring scans.

### Key Analysis Metrics

The dashboard can display metrics such as:

- Total Products Detected
- Misplaced Items
- Empty Shelf Zones
- Low-Stock Zones
- Shelf Health Score

---

## 🏪 Applications / Use Cases

The Retail Shelf Monitoring system can be used in different retail and inventory-management scenarios:

- 🛒 **Supermarkets & Retail Stores** — Monitor shelf conditions and product availability.
- 📦 **Inventory Monitoring** — Identify low-stock and empty shelf areas.
- 🔍 **Shelf Compliance** — Check whether products are placed in their expected shelf zones.
- ⚠️ **Misplacement Detection** — Help identify products that may be placed in incorrect locations.
- 📊 **Store Analytics** — Generate data-driven insights from shelf images.
- 🕒 **Routine Shelf Auditing** — Support regular visual inspection of retail shelves.

---

## 🚀 Future Scope

The project can be further enhanced with the following improvements:

- 🎥 **Real-Time Shelf Monitoring** using live camera feeds.
- 🧠 **Custom YOLO Model Training** for more retail product categories.
- 📱 **Mobile-Friendly Dashboard** for monitoring from different devices.
- ☁️ **Cloud Deployment** for remote access and centralized monitoring.
- 🔔 **Automated Notifications** for critical stock and shelf conditions.
- 📈 **Advanced Retail Analytics** for tracking shelf conditions over time.
- 🏷️ **Product Recognition** for identifying specific products and categories.
- 🔄 **Automated Inventory Integration** with retail inventory systems.

---

## 📚 Project Documentation

Detailed project documents are available in the [`docs`](./docs) folder.

| Document | Description |
|---|---|
| 📄 PRD | Product Requirements Document |
| 📄 FRD | Functional Requirements Document |
| 📄 TRD | Technical Requirements Document |
| 📄 Synopsis | Project synopsis and overview |
| 🎓 College Prompt | College project requirements |

These documents provide detailed information about the project requirements, functionality, technical design, implementation, and academic evaluation.
  
---

## Database Schema (SQLite)

As specified in TRD Section 9, the database `data/shelf_monitoring.db` maintains three primary relational tables:

```sql
-- 1. Shelf Zones Configuration Table
CREATE TABLE zones (
    zone_id TEXT PRIMARY KEY,
    zone_name TEXT NOT NULL,
    expected_category TEXT NOT NULL,
    x_min REAL NOT NULL,
    y_min REAL NOT NULL,
    x_max REAL NOT NULL,
    y_max REAL NOT NULL,
    capacity INTEGER DEFAULT 8
);

-- 2. Detections Log Table
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_w REAL NOT NULL,
    bbox_h REAL NOT NULL,
    zone_id TEXT,
    is_misplaced INTEGER DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (zone_id) REFERENCES zones (zone_id)
);

-- 3. Consolidated Reports Table
CREATE TABLE reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL,
    total_products INTEGER NOT NULL,
    empty_zones_count INTEGER NOT NULL,
    misplaced_count INTEGER NOT NULL,
    shelf_status TEXT NOT NULL,
    health_score REAL NOT NULL,
    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## Model Training & Custom Datasets

To fine-tune YOLOv8n on public datasets (e.g., SKU-110K, Grocery Store Dataset, or Roboflow):

1. Organize your labeled dataset into YOLO format:
   ```text
   dataset/
   ??? images/
   ?   ??? train/
   ?   ??? val/
   ??? labels/
       ??? train/
       ??? val/
   ```
2. Create `dataset.yaml`:
   ```yaml
   path: ./dataset
   train: images/train
   val: images/val
   names:
     0: bottle
     1: can
     2: box
   ```
3. Run training:
   ```bash
   python scripts/train.py --data dataset.yaml --epochs 50 --batch 16
   ```

---

## College Evaluation & Viva Voce Q&A

### Q1: Why was YOLOv8n chosen over other architectures like Faster R-CNN or SSD?
**Answer:** YOLOv8n (nano variant) achieves single-stage real-time inference (approximately 1?3 seconds per frame on standard laptop CPU) with a very small model footprint (~6.5MB) while maintaining high mean Average Precision (mAP@0.5 > 0.6). Two-stage detectors like Faster R-CNN are computationally prohibitive for local execution without dedicated high-end GPUs.

### Q2: How does the system detect empty spaces on a shelf?
**Answer:** Empty space detection utilizes a combination of zone occupancy modeling and spatial gap analysis. The system compares the detected product count against configured shelf capacity and evaluates horizontal bounding box intervals. Gaps between items exceeding the normal threshold are localized, flagged, and visualized with bounding boxes.

### Q3: How is a misplaced product detected?
**Answer:** The store planogram specifies the designated product category for each shelf tier (e.g., Top Shelf: Beverages/Bottles). When a product is detected, its bounding box centroid is mapped to a spatial shelf zone. If the detected item class does not match the zone's expected category (or authorized synonyms), the system flags `is_misplaced = True`, color-codes the box in red, and emits a warning alert.

### Q4: How is the Shelf Health Score calculated?
**Answer:** The health score starts at 100% and applies weighted operational penalties:
- Empty shelf zone: -25 points
- Low stock zone: -15 points
- Misplaced item: -8 points
Status is categorized into:
- **Good**: ? 80%
- **Needs Attention**: 50% ? 79.9%
- **Critical**: < 50%

### Q5: How can this system scale in a real supermarket deployment?
**Answer:** In production, the system can connect to stationary CCTV RTSP camera feeds, run scheduled batch inferences every 15?30 minutes, automatically dispatch restocking notifications to store employees' handheld devices, and push inventory counts directly to the enterprise ERP/POS database.
