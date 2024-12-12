import pandas as pd
import psutil
import random
import time

# Generate synthetic data
num_rows = 50000
data = []

for _ in range(num_rows):
    timestamp = time.time()
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    network = psutil.net_io_counters()
    
    data.append([
        timestamp,
        cpu_usage,
        memory.used / (1024 * 1024),
        disk.read_bytes / (1024 * 1024),
        network.bytes_sent / 1024 + network.bytes_recv / 1024,
        random.uniform(30, 75),  # Random CPU temperature
        disk.free / (1024 * 1024 * 1024),
        disk.used / (1024 * 1024 * 1024),
        random.uniform(1.0, 4.0),  # Random CPU frequency
        memory.available / (1024 * 1024),
        memory.used / (1024 * 1024),
        random.randint(50, 150),  # Random process count
        random.randint(1, 24),  # Random system uptime
        random.uniform(0, 100),  # Random swap usage
        random.uniform(1, 10)  # Random 1-minute load average
    ])

df = pd.DataFrame(data, columns=[
    "Timestamp", "CPU_Usage", "Memory_Usage", "Disk_IO", "Network_Traffic", "CPU_Temperature",
    "Disk_Space_Free", "Disk_Space_Used", "CPU_Frequency", "RAM_Free", "RAM_Used", 
    "Process_Count", "System_Uptime", "Swap_Usage", "Load_Avg"
])

# Save to CSV
df.to_csv("synthetic_server_performance.csv", index=False)
