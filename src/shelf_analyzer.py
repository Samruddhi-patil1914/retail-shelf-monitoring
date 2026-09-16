"""
Shelf Analyzer Module (FR-4, FR-5, FR-6, FR-7)
Implements product counting, zone assignment, misplacement verification,
empty space/gap detection, and overall shelf health scoring.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import sys

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    CATEGORY_SYNONYMS,
    LOW_STOCK_THRESHOLD_PCT,
    CRITICAL_STOCK_THRESHOLD_PCT,
    HEALTH_STATUS_GOOD,
    HEALTH_STATUS_ATTENTION,
    HEALTH_STATUS_CRITICAL
)

class ShelfAnalyzer:
    """Performs retail shelf business logic, compliance checking, and stock analytics."""

    def __init__(self, zones: List[Dict[str, Any]]):
        self.zones = zones

    def analyze(self, image: np.ndarray, detections: List[Dict[str, Any]],
                low_stock_threshold: float = LOW_STOCK_THRESHOLD_PCT) -> Dict[str, Any]:
        """
        Executes end-to-end shelf analysis on detections and image.
        Returns comprehensive analysis dictionary.
        """
        h, w = image.shape[:2]

        # 1. Zone Assignment & Misplacement Detection
        processed_detections, misplaced_items = self._assign_zones_and_check_misplacements(detections)

        # 2. Product Counting
        counts_by_category = self._count_by_category(processed_detections)
        counts_by_zone = self._count_by_zone(processed_detections)
        total_products = len(processed_detections)

        # 3. Empty Space & Low Stock Analysis per Zone
        zone_analytics, empty_slots = self._analyze_zones_stock(
            processed_detections, h, w, low_stock_threshold
        )

        # 4. Shelf Health Score & Overall Status
        empty_count = sum(1 for z in zone_analytics if z['status'] == 'Empty')
        low_stock_count = sum(1 for z in zone_analytics if z['status'] == 'Low Stock')
        misplaced_count = len(misplaced_items)

        health_score, shelf_status = self._compute_shelf_health(
            total_products, len(self.zones), empty_count, low_stock_count, misplaced_count
        )

        # 5. Alert Generation
        alerts = self._generate_alerts(zone_analytics, misplaced_items, shelf_status)

        return {
            'detections': processed_detections,
            'misplaced_items': misplaced_items,
            'counts_by_category': counts_by_category,
            'counts_by_zone': counts_by_zone,
            'total_products': total_products,
            'zone_analytics': zone_analytics,
            'empty_slots': empty_slots,
            'empty_zones_count': empty_count,
            'low_stock_zones_count': low_stock_count,
            'misplaced_count': misplaced_count,
            'health_score': health_score,
            'shelf_status': shelf_status,
            'alerts': alerts
        }

    def _assign_zones_and_check_misplacements(self, detections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Assigns each detection to a shelf zone and checks for category misplacement."""
        processed = []
        misplaced = []

        for det in detections:
            item = dict(det)
            norm_box = item.get('bbox_norm', [0, 0, 0, 0])
            cx = (norm_box[0] + norm_box[2]) / 2.0
            cy = (norm_box[1] + norm_box[3]) / 2.0

            assigned_zone = None
            for zone in self.zones:
                if (zone['x_min'] <= cx <= zone['x_max']) and (zone['y_min'] <= cy <= zone['y_max']):
                    assigned_zone = zone
                    break

            if assigned_zone is None and self.zones:
                assigned_zone = min(self.zones, key=lambda z: abs(((z['y_min'] + z['y_max']) / 2.0) - cy))

            if assigned_zone:
                item['zone_id'] = assigned_zone['zone_id']
                item['zone_name'] = assigned_zone['zone_name']
                expected_cat = assigned_zone['expected_category'].lower()
                detected_cat = item['class_name'].lower()

                # Check compatibility with synonyms
                is_match = self._is_category_compatible(detected_cat, expected_cat)
                item['is_misplaced'] = not is_match
                item['expected_category'] = expected_cat

                if item['is_misplaced']:
                    misplaced.append(item)
            else:
                item['zone_id'] = 'UNKNOWN'
                item['zone_name'] = 'Unassigned Area'
                item['is_misplaced'] = False
                item['expected_category'] = 'any'

            processed.append(item)

        return processed, misplaced

    def _is_category_compatible(self, detected_cat: str, expected_cat: str) -> bool:
        """Determines if detected product fits the expected shelf zone category."""
        if detected_cat == expected_cat:
            return True
        synonyms = CATEGORY_SYNONYMS.get(expected_cat, [expected_cat])
        return any(syn in detected_cat or detected_cat in syn for syn in synonyms)

    def _count_by_category(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarizes product counts per detected category."""
        counts = {}
        for d in detections:
            cat = d['class_name']
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _count_by_zone(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarizes product counts per shelf zone."""
        counts = {z['zone_id']: 0 for z in self.zones}
        for d in detections:
            zid = d.get('zone_id', 'UNKNOWN')
            counts[zid] = counts.get(zid, 0) + 1
        return counts

    def _analyze_zones_stock(self, detections: List[Dict[str, Any]], img_h: int, img_w: int,
                             low_stock_thresh: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Calculates stock level, empty space percentages, and detects visual gap areas."""
        zone_analytics = []
        empty_slots = []

        for zone in self.zones:
            zid = zone['zone_id']
            capacity = zone.get('capacity', 8)
            expected = zone.get('expected_category', 'general')

            zone_items = [d for d in detections if d.get('zone_id') == zid]
            item_count = len(zone_items)

            stock_pct = min(100.0, round((item_count / max(1, capacity)) * 100.0, 1))
            empty_pct = max(0.0, round(100.0 - stock_pct, 1))

            if stock_pct < CRITICAL_STOCK_THRESHOLD_PCT:
                status = 'Empty'
            elif stock_pct < low_stock_thresh:
                status = 'Low Stock'
            else:
                status = 'Well Stocked'

            gap_slots = self._find_zone_gaps(zone, zone_items, img_w, img_h)
            empty_slots.extend(gap_slots)

            zone_analytics.append({
                'zone_id': zid,
                'zone_name': zone['zone_name'],
                'expected_category': expected,
                'current_count': item_count,
                'capacity': capacity,
                'stock_percentage': stock_pct,
                'empty_percentage': empty_pct,
                'status': status,
                'misplaced_items_count': sum(1 for d in zone_items if d.get('is_misplaced', False)),
                'bounds_norm': [zone['x_min'], zone['y_min'], zone['x_max'], zone['y_max']]
            })

        return zone_analytics, empty_slots

    def _find_zone_gaps(self, zone: Dict[str, Any], items: List[Dict[str, Any]],
                        img_w: int, img_h: int) -> List[Dict[str, Any]]:
        """Heuristically finds empty segments along the shelf tier."""
        gaps = []
        zx1, zy1 = int(zone['x_min'] * img_w), int(zone['y_min'] * img_h)
        zx2, zy2 = int(zone['x_max'] * img_w), int(zone['y_max'] * img_h)
        zone_w = zx2 - zx1

        if len(items) == 0:
            gaps.append({
                'zone_id': zone['zone_id'],
                'bbox': [zx1 + 20, zy1 + 15, zx2 - 20, zy2 - 15],
                'label': f"Empty {zone['zone_name']}"
            })
            return gaps

        sorted_items = sorted(items, key=lambda it: it['bbox'][0])

        # Left gap
        first_x1 = sorted_items[0]['bbox'][0]
        if (first_x1 - zx1) > (zone_w * 0.22):
            gaps.append({
                'zone_id': zone['zone_id'],
                'bbox': [zx1 + 10, zy1 + 15, first_x1 - 10, zy2 - 15],
                'label': 'Shelf Gap'
            })

        # Inter-item gaps
        for i in range(len(sorted_items) - 1):
            curr_x2 = sorted_items[i]['bbox'][2]
            next_x1 = sorted_items[i + 1]['bbox'][0]
            gap_w = next_x1 - curr_x2
            if gap_w > (zone_w * 0.18):
                gaps.append({
                    'zone_id': zone['zone_id'],
                    'bbox': [curr_x2 + 5, zy1 + 15, next_x1 - 5, zy2 - 15],
                    'label': 'Stock Gap'
                })

        # Right gap
        last_x2 = sorted_items[-1]['bbox'][2]
        if (zx2 - last_x2) > (zone_w * 0.22):
            gaps.append({
                'zone_id': zone['zone_id'],
                'bbox': [last_x2 + 10, zy1 + 15, zx2 - 10, zy2 - 15],
                'label': 'Shelf Gap'
            })

        return gaps

    def _compute_shelf_health(self, total_products: int, total_zones: int,
                             empty_count: int, low_stock_count: int,
                             misplaced_count: int) -> Tuple[float, str]:
        """Computes composite shelf health score (0 to 100) and qualitative status."""
        score = 100.0
        score -= (empty_count * 25.0)
        score -= (low_stock_count * 15.0)
        score -= (misplaced_count * 8.0)

        if total_products == 0:
            score = 0.0

        score = max(0.0, min(100.0, round(score, 1)))

        if score >= 80.0:
            status = HEALTH_STATUS_GOOD
        elif score >= 50.0:
            status = HEALTH_STATUS_ATTENTION
        else:
            status = HEALTH_STATUS_CRITICAL

        return score, status

    def _generate_alerts(self, zone_analytics: List[Dict[str, Any]],
                        misplaced_items: List[Dict[str, Any]],
                        shelf_status: str) -> List[Dict[str, str]]:
        """Creates actionable notifications for retail personnel."""
        alerts = []

        for za in zone_analytics:
            if za['status'] == 'Empty':
                alerts.append({
                    'severity': 'CRITICAL',
                    'title': f"Empty Shelf: {za['zone_name']}",
                    'message': f"Immediate restocking required. Current stock: {za['current_count']}/{za['capacity']} ({za['stock_percentage']}%)."
                })
            elif za['status'] == 'Low Stock':
                alerts.append({
                    'severity': 'WARNING',
                    'title': f"Low Stock: {za['zone_name']}",
                    'message': f"Stock level below threshold ({za['stock_percentage']}%). Restock recommended soon."
                })

        for mis in misplaced_items:
            alerts.append({
                'severity': 'WARNING',
                'title': f"Misplaced Item in {mis.get('zone_name', 'Shelf')}",
                'message': f"Detected '{mis['class_name']}' in a zone designated for '{mis.get('expected_category')}'. Move to proper section."
            })

        if not alerts:
            alerts.append({
                'severity': 'INFO',
                'title': "Shelf in Optimal Condition",
                'message': "All items are correctly organized, with sufficient stock across all shelves."
            })

        return alerts
