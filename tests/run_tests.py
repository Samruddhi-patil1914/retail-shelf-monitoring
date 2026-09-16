"""
Standalone Unit Test Runner (Runs with Python standard library unittest).
No extra dependencies required!
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import SAMPLES_DIR, DB_PATH
from database.db_manager import DatabaseManager
from src.preprocessor import ImagePreprocessor
from src.detector import ShelfProductDetector
from src.shelf_analyzer import ShelfAnalyzer
from src.report_generator import ReportGenerator

class TestRetailShelfMonitoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseManager()
        cls.zones = cls.db.get_all_zones()
        cls.detector = ShelfProductDetector()
        cls.analyzer = ShelfAnalyzer(cls.zones)

    def test_database_initialization(self):
        self.assertGreaterEqual(len(self.zones), 3)
        zone_ids = [z['zone_id'] for z in self.zones]
        self.assertIn('ZONE_TOP', zone_ids)
        self.assertIn('ZONE_MID', zone_ids)
        self.assertIn('ZONE_BOT', zone_ids)

    def test_image_preprocessor_validation(self):
        valid, _ = ImagePreprocessor.validate_image_file(SAMPLES_DIR / "shelf_normal.png")
        self.assertTrue(valid)
        invalid, _ = ImagePreprocessor.validate_image_file("nonexistent.pdf")
        self.assertFalse(invalid)

    def test_normal_shelf_analysis(self):
        sample_path = SAMPLES_DIR / "shelf_normal.png"
        original, _, _, _ = ImagePreprocessor.preprocess_pipeline(sample_path)
        dets = self.detector.detect(original)
        results = self.analyzer.analyze(original, dets)

        self.assertGreater(results['total_products'], 15)
        self.assertEqual(results['misplaced_count'], 0)
        self.assertGreaterEqual(results['health_score'], 80.0)
        self.assertEqual(results['shelf_status'], 'Good')

    def test_misplaced_shelf_analysis(self):
        sample_path = SAMPLES_DIR / "shelf_misplaced.png"
        original, _, _, _ = ImagePreprocessor.preprocess_pipeline(sample_path)
        dets = self.detector.detect(original)
        results = self.analyzer.analyze(original, dets)

        self.assertGreater(results['misplaced_count'], 0)
        misplaced_classes = [m['class_name'] for m in results['misplaced_items']]
        self.assertTrue('can' in misplaced_classes or 'bottle' in misplaced_classes)

    def test_low_stock_shelf_analysis(self):
        sample_path = SAMPLES_DIR / "shelf_low_stock.png"
        original, _, _, _ = ImagePreprocessor.preprocess_pipeline(sample_path)
        dets = self.detector.detect(original)
        results = self.analyzer.analyze(original, dets)

        self.assertTrue(results['empty_zones_count'] > 0 or results['low_stock_zones_count'] > 0)
        self.assertLess(results['health_score'], 80.0)

    def test_report_generation(self):
        sample_path = SAMPLES_DIR / "shelf_normal.png"
        original, _, _, _ = ImagePreprocessor.preprocess_pipeline(sample_path)
        dets = self.detector.detect(original)
        results = self.analyzer.analyze(original, dets)

        df = ReportGenerator.generate_detections_dataframe("img_unit_test", results)
        self.assertFalse(df.empty)
        self.assertIn('image_id', df.columns)
        self.assertIn('status', df.columns)

        summary = ReportGenerator.generate_nl_summary(results)
        self.assertIn("Shelf Audit Summary", summary)

if __name__ == '__main__':
    unittest.main()
