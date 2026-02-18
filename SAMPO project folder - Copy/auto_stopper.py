
import time
import json
import subprocess
import os
from pathlib import Path

STATUS_FILE = "c:/Users/Jeff/Desktop/sample_DS/outputs/tuning_status.json"
LOG_FILE = "c:/Users/Jeff/Desktop/sample_DS/outputs/auto_stop.log"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.ctime()}] {msg}\n")
    print(msg)

def get_tuner_pid():
    # Find python process running optuna_tuner
    cmd = 'wmic process where "CommandLine like \'%sampo2.agents.optuna_tuner%\'" get ProcessId'
    try:
        output = subprocess.check_output(cmd, shell=True).decode()
        lines = output.strip().split('\n')
        if len(lines) > 1:
            pid = lines[1].strip()
            if pid:
                return int(pid)
    except Exception as e:
        log(f"Error finding PID: {e}")
    return None

def main():
    log("Auto-stopper started. Monitoring for completion of Trial 1 (Index 1).")
    
    while True:
        try:
            if Path(STATUS_FILE).exists():
                with open(STATUS_FILE, 'r') as f:
                    status = json.load(f)
                
                trial = status.get('trial', 0)
                progress = status.get('progress', 0)
                
                # Check if Trial 1 (User's Trial 2) is done or close to done
                if trial == 1 and progress >= 0.99:
                    log(f"Trial 1 is {progress*100:.1f}% complete. Waiting 120s for cleanup/logging.")
                    time.sleep(120)
                    
                    pid = get_tuner_pid()
                    if pid:
                        log(f"Killing Tuner PID: {pid}")
                        os.system(f"taskkill /PID {pid} /F")
                        log("Tuner terminated successfully.")
                        break
                    else:
                        log("Could not find Tuner PID. It might have already exited.")
                        break
                
                if trial > 1:
                    log("Trial 2 (Index 2) seemingly started already. Killing immediately.")
                    pid = get_tuner_pid()
                    if pid:
                        os.system(f"taskkill /PID {pid} /F")
                    break
                    
            time.sleep(10)
            
        except Exception as e:
            log(f"Monitor error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
