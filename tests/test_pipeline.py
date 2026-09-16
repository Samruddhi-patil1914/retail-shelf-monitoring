"""
Automated Unit and Integration Test Suite
Validates database management, preprocessing, detection, shelf analysis, and reporting.
"""

import pytest
import numpy as np
from pathlib import Path
import os
import sys

# Ensure root directory is on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import SAMPLES_DIR, DB_PATH
from database.db_manager import DatabaseManager
from src.preprocessor import ImagePreprocessor
from src.detector import ShelfProductDetector
from src.shelf_analyzer import ShelfAnalyzer
from src.report_generator import ReportGenerator

@pytest.fixture(scope="module")
def db():
    manager = DatabaseManager()
    return manager

@pytest.fixture(scope="module")
def zones(db):
    return db.get_all_zones()

@pytest.fixture(scope="module")
def detector():
    return ShelfProductDetector()

@pytest.fixture(scope="module")
def analyzer(zones):
    return ShelfAnalyzer(zones)

def test_database_initialization(db):
    """Verify database tables and seeded shelf zones."""
    zones = db.get_all_zones()
    assert len(zones) >= 3, "Database should contain at least 3 shelf zones"
    zone_ids = [z['zone_id'] for z in zones]
    assert 'ZONE_TOP' in zone_ids
    assert 'ZONE_MID' in zone_ids
    assert 'ZONE_BOT' in zone_ids

def test_image_preprocessor_validation():
    """Test format validation for valid and invalid paths."""
    valid, msg = ImagePreprocessor.validate_image_file("data/samples/shelf_normal.png")
    assert valid is True

    valid_bad, msg_bad = ImagePreprocessor.validate_image_file("non_existent_file.pdf")
    assert valid_bad is False

def test_normal_shelf_analysis(analyzer, detector):
    """Verify normal shelf produces high health score and zero misplacements."""
    sample_path = SAMPLES_DIR / "shelf_normal.png"
    assert sample_path.exists(), "Sample normal image must exist"

    original, _, _, _ = ImagePreprocessor.preprocess_pipeline(sample_path)
    dets = detector.detect(original)
    results = analyzer.analyze(original, dets)

    assert results['total_products'] > 15, "Normal shelf should have at least 15 items"
    assert results['misplaced_count'] == 0, "Normal shelf should have zero misplaced items"
    assert results['health_score'] >= 80.0, "Health score should be >= 80"
    assert results['shelf_status'] == 'Good'

def test_misplaced_shelf_analysis(analyzer, detector):
    """Verify shelf with misplaced products detects violations."""
    sample_path = SAMPLES_DIR / "shelf_misplaced.png"
    assert sample_path.exists(), "Sample misplaced image must exist"

    original, _, _, _ = ImagePreprocessor.preprocess_pipeline(sample_path)
    dets = detector.detect(original)
    results = analyzer.analyze(original, dets)

    assert results['misplaced_count'] > 0, "Misplaced shelf should flag misplaced items"
    misplaced_names = [m['class_name'] for m in results['misplaced_items']]
    assert len(misplaced_names) >= 1

def test_low_stock_shelf_analysis(analyzer, detector):
    """Verify low-stock shelf detects empty or low-stock zones."""
    sample_path = SAMPLES_DIR / "shelf_low_stock.png"
    assert sample_path.exists(), "Sample low stock image must exist"

    original, _, _, _ = ImagePreprocessor.preprocess_pipeline(sample_path)
    dets = detector.detect(original)
    results = analyzer.analyze(original, dets)

    assert results['empty_zones_count'] > 0 or results['low_stock_zones_count'] > 0
    assert results['health_score'] < 80.0, "Health score should reflect stock deficit"

def test_report_generation(analyzer, detector):
    """Verify CSV export dataframe generation."""
    sample_path = SAMPLES_DIR / "shelf_normal.png"
    original, _, _, _ = ImagePreprocessor.preprocess_pipeline(sample_path)
    dets = detector.detect(original)
    results = analyzer.analyze(original, dets)

    df = ReportGenerator.generate_detections_dataframe("test_img_001", results)
    assert not df.empty
    assert 'image_id' in df.columns
    assert 'class' in df.columns
    assert 'status' in df.columns

    summary = ReportGenerator.generate_nl_summary(results)
    assert "Shelf Audit Summary" in summary
