
import json
import time
import sys
from pathlib import Path

STATUS_FILE = "c:/Users/Jeff/Desktop/sample_DS/outputs/tuning_status.json"

def main():
    print("Waiting for tuning status...")
    last_step = -1
    
    while True:
        try:
            if Path(STATUS_FILE).exists():
                with open(STATUS_FILE, 'r') as f:
                    status = json.load(f)
                
                trial = status.get('trial', 0)
                current = status.get('current_step', 0)
                total = status.get('total_steps', 1)
                progress = status.get('progress', 0)
                
                bar_length = 50
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                
                sys.stdout.write(f"\rTrial {trial+1}: [{bar}] {progress*100:.1f}% ({current}/{total} steps)")
                sys.stdout.flush()
                
                if current >= total:
                    print(f"\nTrial {trial+1} Complete.")
                    # Wait a bit for next trial to start writing
                    time.sleep(5) 
            
            time.sleep(1)
            
        except json.JSONDecodeError:
            pass # Creating file
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
