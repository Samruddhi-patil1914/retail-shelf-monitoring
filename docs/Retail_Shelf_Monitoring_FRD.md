# Functional Requirements Document (FRD)
## Project: Retail Shelf Monitoring Using Computer Vision and Data Science

---

## 1. Purpose

This document defines the functional requirements for the Retail Shelf Monitoring system — an application that uses computer vision to analyze shelf images, detect and count products, identify empty/low-stock areas, flag misplaced items, and present results via a dashboard.

---

## 2. Scope

The system covers: image input, product detection, product counting, empty-shelf detection, misplacement detection, alert generation, and dashboard visualization/reporting. It does not cover real-time CCTV integration, POS/inventory system integration, or multi-store analytics (these are future enhancements).

---

## 3. Actors

| Actor | Description |
|---|---|
| Store Employee/User | Uploads shelf image, views results, exports reports |
| System/Admin (optional) | Manages product categories, retrains/updates model |
| System (automated) | Runs detection pipeline, generates alerts |

---

## 4. Functional Requirements

### FR-1: Image Input
- **FR-1.1** System shall allow the user to upload a shelf image (JPG/PNG) via the dashboard.
- **FR-1.2** System shall validate the uploaded file format and size before processing.
- **FR-1.3** System shall display the uploaded image on the dashboard for reference.

### FR-2: Image Preprocessing
- **FR-2.1** System shall resize the uploaded image to the model's required input dimensions.
- **FR-2.2** System shall normalize pixel values before passing the image to the detection model.

### FR-3: Product Detection
- **FR-3.1** System shall run the object detection model (YOLOv8n) on the preprocessed image.
- **FR-3.2** System shall return bounding boxes, class labels, and confidence scores for each detected product.
- **FR-3.3** System shall discard detections below a configurable confidence threshold.
- **FR-3.4** System shall overlay bounding boxes and labels on the displayed image.

### FR-4: Product Counting
- **FR-4.1** System shall count the number of detected products per category/class.
- **FR-4.2** System shall display a summary table of product counts by category.

### FR-5: Empty/Low-Stock Detection
- **FR-5.1** System shall analyze shelf regions without detected products to identify empty spaces.
- **FR-5.2** System shall calculate a "stock level" or "empty-space percentage" per shelf zone.
- **FR-5.3** System shall flag any zone below a configurable stock threshold as "low stock" or "empty."

### FR-6: Misplaced Product Detection
- **FR-6.1** System shall compare detected product categories against the expected category for each shelf zone.
- **FR-6.2** System shall flag products detected in an incorrect zone as "misplaced."
- **FR-6.3** System shall highlight misplaced items visually (e.g., different bounding box color).

### FR-7: Alerts & Shelf Status
- **FR-7.1** System shall generate a list of alerts (empty zones, low stock, misplaced items) after analysis.
- **FR-7.2** System shall compute an overall shelf-health score/status (e.g., Good / Needs Attention / Critical).

### FR-8: Dashboard & Visualization
- **FR-8.1** System shall display the annotated image, product counts, and alerts in a single dashboard view.
- **FR-8.2** System shall provide charts (bar/pie) summarizing product distribution and stock status.
- **FR-8.3** Dashboard shall update results dynamically when a new image is uploaded.

### FR-9: Reporting & Export
- **FR-9.1** System shall allow the user to export detection results as a CSV file.
- **FR-9.2** System shall allow the user to download the annotated image.

### FR-10: (Optional) Natural-Language Summary
- **FR-10.1** System may generate a text summary of shelf status using a local LLM (e.g., "Zone B is 40% empty; 3 items misplaced").

---

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | Detection + analysis should complete within a few seconds per image on a standard laptop CPU/GPU |
| Usability | Dashboard should be simple and require no technical training to operate |
| Portability | System should run locally without requiring paid cloud services |
| Reliability | System should handle invalid/corrupt image uploads gracefully with an error message |
| Scalability (future) | Architecture should allow extension to batch/multi-image or real-time video processing |

---

## 6. Assumptions & Constraints

- Model accuracy depends on the quality and size of the training dataset used.
- System is designed for single-image analysis in the MVP; not real-time video by default.
- All tools/models used are free and open-source, runnable on a standard laptop.
- Product category zones (for misplacement detection) must be predefined/configured.

---

## 7. Dependencies

- Pretrained/fine-tuned YOLOv8n model weights
- Annotated dataset for training and validation
- Python environment with OpenCV, Ultralytics, Streamlit, Pandas installed
