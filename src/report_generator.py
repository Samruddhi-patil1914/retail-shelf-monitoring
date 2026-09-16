"""
Reporting and Export Module (FR-9, FR-10)
Generates structured CSV exports, analytics summaries, and automated natural-language diagnosis.
Supports local Ollama LLM execution with graceful rule-based fallback.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import requests
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EXPORTS_DIR

class ReportGenerator:
    """Produces downloadable CSV reports and natural-language summaries."""

    @staticmethod
    def generate_detections_dataframe(image_id: str, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Builds DataFrame following TRD Section 6.3 output specification:
        fields: image_id, class, confidence, bounding_box, zone, status
        """
        rows = []
        for det in analysis_results.get('detections', []):
            x, y, w, h = det.get('bbox_xywh', [0, 0, 0, 0])
            status = "MISPLACED" if det.get('is_misplaced') else "CORRECT"

            rows.append({
                'image_id': image_id,
                'class': det.get('class_name', 'unknown'),
                'confidence': det.get('confidence', 0.0),
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h,
                'zone': det.get('zone_name', 'Unassigned'),
                'status': status
            })

        return pd.DataFrame(rows)

    @classmethod
    def export_csv_report(cls, image_id: str, analysis_results: Dict[str, Any],
                          output_path: Optional[Path] = None) -> Path:
        """Saves detection results to a standardized CSV file."""
        df = cls.generate_detections_dataframe(image_id, analysis_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = output_path or (EXPORTS_DIR / f"shelf_report_{image_id}_{timestamp}.csv")

        df.to_csv(str(file_path), index=False)
        return file_path

    @staticmethod
    def generate_nl_summary(analysis_results: Dict[str, Any], use_ollama: bool = False,
                            ollama_model: str = "llama3.2:3b") -> str:
        """
        Generates professional natural-language summary for store managers (FR-10).
        Attempts local Ollama endpoint if requested, otherwise uses expert rule-based generator.
        """
        if use_ollama:
            llm_text = ReportGenerator._query_ollama(analysis_results, ollama_model)
            if llm_text:
                return llm_text

        # Expert rule-based generator fallback (100% reliable, zero external dependencies)
        return ReportGenerator._generate_rule_based_summary(analysis_results)

    @staticmethod
    def _generate_rule_based_summary(data: Dict[str, Any]) -> str:
        """Produces clear, executive-level natural-language audit report."""
        total = data.get('total_products', 0)
        misplaced = data.get('misplaced_count', 0)
        status = data.get('shelf_status', 'Unknown')
        score = data.get('health_score', 0.0)
        zones = data.get('zone_analytics', [])

        lines = [
            f"### ?? Shelf Audit Summary",
            f"**Overall Assessment:** The monitored shelf is currently classified as **{status.upper()}** with a health score of **{score}/100**.",
            f"- **Product Volume:** A total of **{total} products** were detected across all shelf sections.",
            f"- **Arrangement Quality:** **{misplaced} item(s)** were identified as misplaced or in non-designated zones."
        ]

        # Zone breakdowns
        lines.append("\n**Shelf Zone Diagnostics:**")
        for z in zones:
            z_status = z['status']
            icon = "?" if z_status == "Well Stocked" else ("??" if z_status == "Low Stock" else "??")
            mis_text = f", including {z['misplaced_items_count']} misplaced item(s)" if z['misplaced_items_count'] > 0 else ""
            lines.append(
                f"- {icon} **{z['zone_name']}:** {z['current_count']}/{z['capacity']} items ({z['stock_percentage']}% capacity) ? "
                f"Status: **{z_status}**{mis_text}."
            )

        # Actionable recommendations
        lines.append("\n**Recommended Action Items:**")
        if misplaced > 0:
            lines.append(f"1. Dispatch staff to return {misplaced} misplaced product(s) to their respective designated aisles.")
        empty_zones = [z['zone_name'] for z in zones if z['status'] == 'Empty']
        low_zones = [z['zone_name'] for z in zones if z['status'] == 'Low Stock']

        if empty_zones:
            lines.append(f"2. Urgent replenishment needed for empty shelves: {', '.join(empty_zones)}.")
        elif low_zones:
            lines.append(f"2. Schedule restock replenishment for low shelves: {', '.join(low_zones)}.")

        if not empty_zones and not low_zones and misplaced == 0:
            lines.append("? No immediate operational intervention needed. Shelf satisfies store merchandising standards.")

        return "\n".join(lines)

    @staticmethod
    def _query_ollama(data: Dict[str, Any], model_name: str) -> Optional[str]:
        """Queries local Ollama instance if available on localhost:11434."""
        try:
            prompt = (
                f"You are an AI retail operations assistant. Generate a concise 3-paragraph "
                f"shelf health summary based on these scan results:\n"
                f"Total Products: {data.get('total_products')}\n"
                f"Misplaced Items: {data.get('misplaced_count')}\n"
                f"Shelf Health: {data.get('shelf_status')} ({data.get('health_score')}/100)\n"
                f"Zones: {json.dumps(data.get('zone_analytics'))}\n"
            )
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=4.0
            )
            if resp.status_code == 200:
                return resp.json().get('response', '')
        except Exception:
            pass
        return None
