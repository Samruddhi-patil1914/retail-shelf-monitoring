"""
Database Manager for Retail Shelf Monitoring System
Implements SQLite storage for detections, shelf zones, and historical monitoring reports.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import sys

# Add project root to path if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH, DEFAULT_ZONES

class DatabaseManager:
    """Manages SQLite database operations for retail shelf analysis."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Return a thread-safe connection to the SQLite database."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize tables according to TRD Section 9 specifications."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Zones Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS zones (
                    zone_id TEXT PRIMARY KEY,
                    zone_name TEXT NOT NULL,
                    expected_category TEXT NOT NULL,
                    x_min REAL NOT NULL,
                    y_min REAL NOT NULL,
                    x_max REAL NOT NULL,
                    y_max REAL NOT NULL,
                    capacity INTEGER DEFAULT 8
                )
            ''')

            # Detections Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
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
                )
            ''')

            # Reports Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    total_products INTEGER NOT NULL,
                    empty_zones_count INTEGER NOT NULL,
                    misplaced_count INTEGER NOT NULL,
                    shelf_status TEXT NOT NULL,
                    health_score REAL NOT NULL,
                    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

        # Seed default zones if empty
        self._seed_default_zones()

    def _seed_default_zones(self):
        """Seeds default 3-tier zones if table is empty."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM zones')
            if cursor.fetchone()[0] == 0:
                for zone in DEFAULT_ZONES:
                    cursor.execute('''
                        INSERT INTO zones (zone_id, zone_name, expected_category, x_min, y_min, x_max, y_max, capacity)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        zone['zone_id'],
                        zone['zone_name'],
                        zone['expected_category'],
                        zone['x_min'],
                        zone['y_min'],
                        zone['x_max'],
                        zone['y_max'],
                        zone.get('capacity', 8)
                    ))
                conn.commit()

    def get_all_zones(self) -> List[Dict[str, Any]]:
        """Retrieve all registered shelf zones."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM zones ORDER BY y_min ASC')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_zone(self, zone_id: str, zone_name: str, expected_category: str, capacity: int):
        """Update category or name for an existing zone."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE zones
                SET zone_name = ?, expected_category = ?, capacity = ?
                WHERE zone_id = ?
            ''', (zone_name, expected_category, capacity, zone_id))
            conn.commit()

    def log_detections(self, image_id: str, detections: List[Dict[str, Any]]):
        """Bulk insert detections for a scanned image."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for d in detections:
                cursor.execute('''
                    INSERT INTO detections (
                        image_id, class_name, confidence,
                        bbox_x, bbox_y, bbox_w, bbox_h,
                        zone_id, is_misplaced, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id,
                    d.get('class_name', 'unknown'),
                    float(d.get('confidence', 0.0)),
                    float(d.get('bbox_x', 0.0)),
                    float(d.get('bbox_y', 0.0)),
                    float(d.get('bbox_w', 0.0)),
                    float(d.get('bbox_h', 0.0)),
                    d.get('zone_id'),
                    1 if d.get('is_misplaced', False) else 0,
                    now
                ))
            conn.commit()

    def log_report(self, image_id: str, total_products: int, empty_zones_count: int,
                   misplaced_count: int, shelf_status: str, health_score: float) -> int:
        """Log overall shelf monitoring report."""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO reports (
                    image_id, total_products, empty_zones_count,
                    misplaced_count, shelf_status, health_score, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_id, total_products, empty_zones_count,
                misplaced_count, shelf_status, health_score, now
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_reports(self, limit: int = 15) -> pd.DataFrame:
        """Fetch recent shelf monitoring scan reports as a DataFrame."""
        with self.get_connection() as conn:
            query = '''
                SELECT report_id, image_id, total_products, empty_zones_count,
                       misplaced_count, shelf_status, health_score, generated_at
                FROM reports
                ORDER BY report_id DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df

    def get_detections_for_image(self, image_id: str) -> pd.DataFrame:
        """Fetch all detections associated with a specific image scan."""
        with self.get_connection() as conn:
            query = '''
                SELECT id, image_id, class_name, confidence, bbox_x, bbox_y, bbox_w, bbox_h,
                       zone_id, is_misplaced, timestamp
                FROM detections
                WHERE image_id = ?
            '''
            return pd.read_sql_query(query, conn, params=(image_id,))
