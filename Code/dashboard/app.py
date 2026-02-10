from flask import Flask, render_template, jsonify
import json
import time
from pathlib import Path
import sys

# Setup Path to read config
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
import config

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Jio CCTV Analytics</title>
            <meta http-equiv="refresh" content="1"> <style>
                body { font-family: sans-serif; text-align: center; background: #f0f0f0; }
                .card { background: white; padding: 20px; margin: 20px auto; width: 300px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .metric { font-size: 2em; font-weight: bold; color: #007bff; }
                .occupied { color: #dc3545; }
                .free { color: #28a745; }
            </style>
        </head>
        <body>
            <h1>Live Office Occupancy</h1>
            <div id="dashboard">Loading...</div>
            
            <script>
                async function fetchStats() {
                    let response = await fetch('/api/stats');
                    let data = await response.json();
                    
                    document.getElementById('dashboard').innerHTML = `
                        <div class="card">
                            <h3>Total Utilization</h3>
                            <div class="metric">${data.metrics.utilization.toFixed(1)}%</div>
                        </div>
                        <div class="card">
                            <h3>Occupied Chairs</h3>
                            <div class="metric occupied">${data.metrics.occupied}</div>
                        </div>
                        <div class="card">
                            <h3>Available Chairs</h3>
                            <div class="metric free">${data.metrics.empty}</div>
                        </div>
                    `;
                }
                fetchStats();
            </script>
        </body>
    </html>
    """

@app.route('/api/stats')
def get_stats():
    try:
        with open(config.LIVE_STATE_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({"metrics": {"utilization": 0, "occupied": 0, "empty": 0}})

if __name__ == "__main__":
    print(f"[INFO] Dashboard running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)