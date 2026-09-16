# Product Requirements Document (PRD)
## Project: Retail Shelf Monitoring Using Computer Vision and Data Science

---

## 1. Problem Statement & Objective

**Problem:** Manual shelf monitoring in retail stores is time-consuming, error-prone, and slow to catch empty or misplaced products, leading to lost sales and poor customer experience.

**Objective:** Build a computer-vision-based system that analyzes shelf images to detect products, count stock, flag empty/low-stock zones, identify misplaced items, and present results on a simple dashboard for store staff.

---

## 2. Target Users

- Retail store employees / shelf-stocking staff
- Store managers monitoring inventory health
- Small/medium retail chains without expensive existing shelf-analytics tools
- (Demo audience) College evaluators/faculty assessing the AI/CV pipeline

---

## 3. Key Features / Functional Requirements

1. Upload or capture a shelf image.
2. Detect and localize products using object detection (bounding boxes).
3. Count products per detected category/class.
4. Detect empty/low-stock shelf segments (gaps in shelf image).
5. Flag misplaced products (wrong shelf/category zone).
6. Generate a shelf-health score/summary.
7. Dashboard to visualize detections, counts, and alerts.
8. Downloadable report (CSV/image with annotations).

---

## 4. System Workflow

```
Image Input (upload/CCTV snapshot)
        ↓
Preprocessing (resize, normalize)
        ↓
Object Detection (YOLO model)
        ↓
   ┌────────────┬─────────────────┬──────────────────┐
Product Count   Empty-Space Detect   Misplacement Check
   └────────────┴─────────────────┴──────────────────┘
        ↓
Shelf Analysis & Alert Generation
        ↓
Dashboard Visualization (Streamlit)
        ↓
Report Export (CSV/annotated image)
```

---

## 5. FREE/Open-Source Tools & Technologies

| Purpose | Tool |
|---|---|
| Language | Python 3.10+ |
| CV processing | OpenCV |
| Object detection | Ultralytics YOLOv8 (open-source, AGPL/free for research) |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Annotation tool | LabelImg / Roboflow (free tier, open-source) |
| IDE | VS Code / Jupyter / Google Colab (free GPU) |
| Version control | Git + GitHub |

---

## 6. FREE/Open-Source LLM Models (optional layer, e.g., generating text summaries/alerts)

| Model | Size | Why |
|---|---|---|
| **Phi-3-mini** (Microsoft) | 3.8B | Very lightweight, runs on CPU/laptop, good for generating natural-language shelf-status summaries |
| **Llama 3.2 3B / 1B** (Meta) | 1B–3B | Free, open weights, small enough for local inference (Ollama), good instruction-following for report text |
| **Gemma 2 2B** (Google) | 2B | Lightweight, easy local deployment via Ollama, decent for short summary generation |

**Recommendation:** Use **Llama 3.2 3B via Ollama** — easiest local setup, good balance of quality/speed on a laptop, only needed if you want auto-generated text alerts (e.g., "Aisle 3 shelf B is 40% empty"). This is an **optional enhancement**, not core to the CV pipeline.

---

## 7. FREE/Open-Source AI/ML Models & Algorithms

| Task | Model/Algorithm |
|---|---|
| Product detection | YOLOv8n/s (nano/small — laptop-friendly) |
| Alternative lightweight detector | MobileNet-SSD (via OpenCV DNN) |
| Empty-space detection | Contour detection (OpenCV) + rule-based gap analysis, or simple CNN classifier |
| Misplacement detection | Template matching / feature matching (ORB, SIFT) or classification with a small CNN (ResNet18) |
| Product counting | Derived directly from YOLO detection outputs (bounding box count per class) |
| Optional clustering (grouping similar products) | K-Means on extracted image features |

**Recommendation:** Start with **YOLOv8n** (nano) — fastest to train/fine-tune on a laptop, smallest model size, sufficient accuracy for a college-scale demo dataset.

---

## 8. Recommended Tech Stack (single primary stack)

- **Frontend/Dashboard:** Streamlit
- **Backend:** Python (FastAPI, only if you need a separate API layer; otherwise Streamlit alone suffices for MVP)
- **Database:** SQLite (lightweight, file-based, zero setup)
- **AI/ML:** YOLOv8n (Ultralytics) + OpenCV for image processing
- **LLM (optional):** Llama 3.2 3B via Ollama (local, free)
- **APIs:** None required (fully offline/local); optional Roboflow API (free tier) for dataset management
- **Deployment:** Localhost demo via Streamlit; optionally free-tier deployment on Streamlit Community Cloud or Hugging Face Spaces

---

## 9. Dataset Requirements & Sources

- **Minimum need:** 200–500 annotated shelf images (bounding boxes per product class) for a workable MVP; more improves accuracy.
- **Sources (free):**
  - **SKU-110K** (large-scale retail shelf dataset, open for research)
  - **Grocery Store Dataset** (Freiburg, open-source on GitHub)
  - **Roboflow Universe** (many public retail/shelf datasets, free to download)
  - Self-collected: photograph shelves at local stores/supermarkets (with permission) for a custom demo set
- **Annotation:** Use LabelImg or Roboflow's free annotation tool to label bounding boxes.

---

## 10. Evaluation Metrics

- **Detection:** mAP (mean Average Precision), Precision, Recall
- **Counting accuracy:** Compare predicted vs. ground-truth product counts (MAE)
- **Empty-shelf detection:** Accuracy/F1-score on labeled empty vs. stocked regions
- **System performance:** Inference time per image (should be near-real-time on laptop CPU/GPU)
- **Usability:** Dashboard clarity (qualitative, via peer/faculty feedback)

---

## 11. MVP Scope

- Upload single shelf image → run YOLOv8n detection → display bounding boxes + counts per category
- Basic empty-space detection using contour/gap heuristics
- Simple Streamlit dashboard showing image, counts, and alert list ("Low stock in Zone X")
- CSV export of detection results

**Out of scope for MVP:** real-time CCTV feed, multi-store dashboard, LLM-generated summaries, misplacement detection (can be a stretch goal).

---

## 12. Future Enhancements

- Real-time monitoring via CCTV/RTSP camera feeds
- Automatic inventory system integration
- Push notifications (email/SMS) to staff
- Product price/label OCR detection
- Sales prediction using historical shelf-stock data
- LLM-based natural-language daily shelf reports
- Multi-camera, multi-store dashboard with historical trend analysis

---

## 13. Feasibility & Estimated Resources

- **Hardware:** A standard laptop (8GB+ RAM) is sufficient for YOLOv8n inference; Google Colab free GPU recommended for training.
- **Team size:** 1–3 students (solo feasible for MVP; team better for full scope incl. misplacement + LLM layer).
- **Timeline (typical TY project):**
  - Weeks 1–2: Dataset collection/annotation
  - Weeks 3–5: Model training (YOLOv8n) & evaluation
  - Weeks 6–7: Empty-space & misplacement logic
  - Weeks 8–9: Dashboard integration (Streamlit)
  - Weeks 10–12: Testing, report writing, buffer
- **Cost:** ₹0 — all tools, models, and datasets are free/open-source; Colab free tier covers training compute.
- **Risk factors:** Limited/imbalanced dataset may reduce detection accuracy — mitigate with data augmentation (Albumentations, free) and transfer learning from pretrained YOLO weights.
