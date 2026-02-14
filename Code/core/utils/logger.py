import csv
import json
import time
from pathlib import Path
import config

class AnalyticsLogger:
    def __init__(self):
        self.csv_path = config.KPI_FILE
        self.json_path = config.LIVE_STATE_FILE
        
        # Initialize CSV Header
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Frame", "Occupied_Count", "Empty_Count", "Total_Chairs"])

    def log_frame(self, frame_id, chair_states, current_fps=0.0):
            occupied = sum(1 for s in chair_states.values() if s == 'OCCUPIED')
            total = len(chair_states)
            empty = total - occupied

            # 1. Update Live JSON
            live_data = {
                "timestamp": time.time(),
                "frame": frame_id,
                "fps": round(current_fps, 1), # Added FPS
                "metrics": {
                    "occupied": occupied,
                    "empty": empty,
                    "utilization": (occupied / total * 100) if total > 0 else 0
                },
                "chairs": chair_states
            }
            
            with open(self.json_path, 'w') as f:
                json.dump(live_data, f)

            # 2. Append to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%H:%M:%S"), frame_id, occupied, empty, total])