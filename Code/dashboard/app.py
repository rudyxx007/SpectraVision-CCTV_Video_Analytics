from flask import Flask, render_template_string, jsonify, Response, request
import json
import time
from pathlib import Path
import sys
import cv2

# Setup Path to read config
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import config

app = Flask(__name__, static_folder='static', static_url_path='/static')

# --- GLOBALS SHARED WITH MAIN.PY ---
latest_frame = None
HEATMAP_ACTIVE = False # Main.py reads this directly to trigger cv2 processing

def update_video_frame(frame):
    """Called by main.py to update the live stream frame."""
    global latest_frame
    latest_frame = frame

def generate_frames():
    """Generator function to stream the video as MJPEG."""
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        
        # Encode the frame as JPEG
        success, buffer = cv2.imencode('.jpg', latest_frame)
        if not success:
            continue
            
        frame_bytes = buffer.tobytes()
        # Yield the frame in the multipart format expected by browsers
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- HTML TEMPLATE (Enterprise Dark Mode) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Jio SpectraVision | Operations Overwatch</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .jio-blue { color: #0033CC; }
        .neon-red { text-shadow: 0 0 10px rgba(239,68,68,0.7); }
        .neon-green { text-shadow: 0 0 10px rgba(34,197,94,0.7); }
        /* Custom scrollbar for alert log */
        #alert-log::-webkit-scrollbar { width: 6px; }
        #alert-log::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    </style>
</head>
<body class="min-h-screen flex flex-col font-sans bg-[#0B1120] text-slate-200">

    <header class="bg-gray-900 border-b border-gray-800 p-4 flex justify-between items-center shadow-lg">
        <div class="flex items-center space-x-3">
            <img src="/static/jio_logo.png" alt="Jio Logo" class="h-8 w-auto bg-white rounded-sm px-1">
            <h1 class="text-xl font-bold tracking-wider">SPECTRA<span class="jio-blue">VISION</span> <span class="text-sm font-normal text-gray-400">| Operations Overwatch</span></h1>
        </div>
        <div class="flex items-center space-x-4">
            <button id="heatmap-btn" onclick="toggleHeatmap()" class="bg-gray-800 hover:bg-gray-700 border border-gray-600 px-4 py-1.5 rounded text-sm font-bold transition-colors">
                🔥 Enable Heatmap
            </button>
            <div class="text-sm text-gray-400 font-mono" id="clock">00:00:00</div>
        </div>
    </header>

    <main class="flex-1 p-6 grid grid-cols-12 gap-6">
        
        <div class="col-span-8 flex flex-col space-y-6">
            
            <div class="grid grid-cols-4 gap-4">
                <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 shadow flex flex-col items-center">
                    <span class="text-gray-400 text-xs uppercase tracking-widest">Total Chairs</span>
                    <span class="text-2xl font-bold mt-1" id="kpi-total">-</span>
                </div>
                <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 shadow flex flex-col items-center">
                    <span class="text-gray-400 text-xs uppercase tracking-widest">Occupied</span>
                    <span class="text-2xl font-bold mt-1 text-red-500 neon-red" id="kpi-occ">-</span>
                </div>
                <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 shadow flex flex-col items-center">
                    <span class="text-gray-400 text-xs uppercase tracking-widest">Available</span>
                    <span class="text-2xl font-bold mt-1 text-green-500 neon-green" id="kpi-empty">-</span>
                </div>
                <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 shadow flex flex-col items-center">
                    <span class="text-gray-400 text-xs uppercase tracking-widest">Utilization</span>
                    <span class="text-2xl font-bold mt-1" id="kpi-util">-</span>
                </div>
            </div>

            <div class="bg-black rounded-lg border border-gray-700 relative flex items-center justify-center overflow-hidden" style="min-height: 600px;">
                <img src="/video_feed" alt="Live CCTV Feed" class="w-full h-full object-contain" id="live-video">
                <div class="absolute top-4 left-4 flex space-x-2">
                    <span class="flex h-3 w-3 relative mt-1">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                    </span>
                    <span class="text-xs text-white font-mono bg-black bg-opacity-50 px-2 py-0.5 rounded">LIVE</span>
                    <span class="text-xs text-blue-400 font-mono bg-black bg-opacity-50 px-2 py-0.5 rounded" id="fps-badge">FPS: --</span>
                </div>
            </div>
        </div>

        <div class="col-span-4 flex flex-col space-y-6">
            
            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 shadow h-64 flex flex-col">
                <h2 class="text-sm text-gray-400 uppercase tracking-widest mb-2">Occupancy Trend (Last 60s)</h2>
                <div class="flex-1 relative w-full h-full">
                    <canvas id="occupancyChart"></canvas>
                </div>
            </div>

            <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 shadow flex-1 flex flex-col max-h-[600px]">
                <h2 class="text-sm text-gray-400 uppercase tracking-widest mb-2 flex justify-between">
                    <span>System Alerts</span>
                    <span class="text-xs bg-gray-700 px-2 rounded-full py-0.5" id="log-count">0 Events</span>
                </h2>
                <div id="alert-log" class="flex-1 overflow-y-auto space-y-2 font-mono text-xs pr-2">
                    <div class="text-blue-400">[SYSTEM] Operations Overwatch Initialized. Awaiting streams...</div>
                </div>
            </div>

        </div>
    </main>

    <script>
        // Clock
        setInterval(() => { document.getElementById('clock').innerText = new Date().toLocaleTimeString(); }, 1000);

        // Chart.js Setup
        const ctx = document.getElementById('occupancyChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(20).fill(''), // X-axis (time)
                datasets: [
                    { label: 'Occupied', data: Array(20).fill(0), borderColor: '#EF4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 },
                    { label: 'Available', data: Array(20).fill(0), borderColor: '#22C55E', borderWidth: 2, borderDash: [5, 5], tension: 0.4, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                animation: { duration: 0 }, // Turn off animation for snappy live updates
                scales: { 
                    y: { beginAtZero: true, grid: { color: '#334155' }, ticks: { color: '#94A3B8', stepSize: 1 } },
                    x: { grid: { display: false } }
                },
                plugins: { legend: { labels: { color: '#94A3B8', boxWidth: 12 } } }
            }
        });

        // Heatmap Toggle Logic
        async function toggleHeatmap() {
            const res = await fetch('/api/toggle_heatmap', {method: 'POST'});
            const data = await res.json();
            const btn = document.getElementById('heatmap-btn');
            if(data.status) {
                btn.innerText = "🛑 Disable Heatmap";
                btn.classList.add('bg-red-900', 'text-red-200', 'border-red-700');
            } else {
                btn.innerText = "🔥 Enable Heatmap";
                btn.classList.remove('bg-red-900', 'text-red-200', 'border-red-700');
            }
        }

        // Fetch Data & Update UI
        let prevOccupied = -1;
        let eventCount = 0;

        async function fetchStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                // Update FPS
                document.getElementById('fps-badge').innerText = `FPS: ${data.fps}`;

                // Update KPIs
                const util = data.metrics.utilization;
                document.getElementById('kpi-total').innerText = data.metrics.occupied + data.metrics.empty;
                document.getElementById('kpi-occ').innerText = data.metrics.occupied;
                document.getElementById('kpi-empty').innerText = data.metrics.empty;
                
                const utilEl = document.getElementById('kpi-util');
                utilEl.innerText = util.toFixed(1) + '%';
                utilEl.className = `text-2xl font-bold mt-1 ${util > 80 ? 'text-red-500 neon-red' : 'text-jio-blue'}`;

                // Update Chart
                chart.data.datasets[0].data.push(data.metrics.occupied);
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.push(data.metrics.empty);
                chart.data.datasets[1].data.shift();
                chart.update();

                // Alert Logic (Log changes)
                if (prevOccupied !== -1 && prevOccupied !== data.metrics.occupied) {
                    const log = document.getElementById('alert-log');
                    const time = new Date().toLocaleTimeString();
                    const msg = data.metrics.occupied > prevOccupied ? 'Occupancy Increased' : 'Occupancy Decreased';
                    const color = data.metrics.occupied > prevOccupied ? 'text-red-400' : 'text-green-400';
                    
                    log.insertAdjacentHTML('afterbegin', `<div class="${color}">[${time}] ${msg}. Current: ${data.metrics.occupied}</div>`);
                    eventCount++;
                    document.getElementById('log-count').innerText = `${eventCount} Events`;
                }
                prevOccupied = data.metrics.occupied;

            } catch (err) { /* Silent fail if server drops briefly */ }
        }

        // Poll every 1 second
        setInterval(fetchStats, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stats')
def get_stats():
    try:
        with open(config.LIVE_STATE_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({"fps": 0, "metrics": {"utilization": 0, "occupied": 0, "empty": 0}})

@app.route('/api/toggle_heatmap', methods=['POST'])
def toggle_heatmap():
    global HEATMAP_ACTIVE
    HEATMAP_ACTIVE = not HEATMAP_ACTIVE
    return jsonify({"status": HEATMAP_ACTIVE})

@app.route('/video_feed')
def video_feed():
    # Returns the streaming response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')