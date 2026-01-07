#!/usr/bin/env python3
"""
Claude Code Study - Submission Watcher
Monitors submissions directory and triggers evaluation for new submissions
"""

import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Set

# Configuration
BASE_DIR = Path(__file__).parent.parent
SUBMISSIONS_DIR = BASE_DIR / "submissions"
EVALUATIONS_DIR = BASE_DIR / "evaluations"

POLL_INTERVAL = 60  # seconds


def get_pending_submissions() -> list:
    """Find submissions that need evaluation."""
    pending = []
    
    for week in range(1, 6):
        submissions_dir = SUBMISSIONS_DIR / f"week{week}"
        evaluations_dir = EVALUATIONS_DIR / f"week{week}"
        
        if not submissions_dir.exists():
            continue
        
        for participant_dir in submissions_dir.iterdir():
            if not participant_dir.is_dir():
                continue
            
            participant_id = participant_dir.name
            metadata_file = participant_dir / "metadata.json"
            eval_file = evaluations_dir / f"{participant_id}.json"
            
            # Check if submitted but not evaluated
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                status = metadata.get("status", "")
                
                # Only evaluate if cloned and not yet evaluated
                if status in ["cloned", "submitted"] and not eval_file.exists():
                    pending.append({
                        "week": week,
                        "participant_id": participant_id,
                        "submitted_at": metadata.get("submitted_at")
                    })
    
    return pending


def run_evaluation(week: int, participant_id: str) -> bool:
    """Run evaluation for a submission."""
    print(f"  Evaluating week{week}/{participant_id}...")
    
    try:
        result = subprocess.run(
            [
                "python3",
                str(BASE_DIR / "backend" / "evaluator.py"),
                "evaluate",
                str(week),
                participant_id
            ],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"  ✅ Completed")
            return True
        else:
            print(f"  ❌ Failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ Timeout")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def update_submission_status(week: int, participant_id: str, status: str):
    """Update submission metadata status."""
    metadata_file = SUBMISSIONS_DIR / f"week{week}" / participant_id / "metadata.json"
    
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        metadata["status"] = status
        metadata["evaluated_at"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def watch_loop():
    """Main watch loop."""
    print("=" * 50)
    print("Claude Code Study - Submission Watcher")
    print("=" * 50)
    print(f"Watching: {SUBMISSIONS_DIR}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print("-" * 50)
    
    while True:
        try:
            pending = get_pending_submissions()
            
            if pending:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(pending)} pending submissions")
                
                for submission in pending:
                    week = submission["week"]
                    participant_id = submission["participant_id"]
                    
                    success = run_evaluation(week, participant_id)
                    
                    if success:
                        update_submission_status(week, participant_id, "evaluated")
                    else:
                        update_submission_status(week, participant_id, "evaluation_failed")
            
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\nWatcher stopped.")
            break
        except Exception as e:
            print(f"Error in watch loop: {e}")
            time.sleep(POLL_INTERVAL)


def run_once():
    """Run evaluation once for all pending submissions."""
    pending = get_pending_submissions()
    
    print(f"Found {len(pending)} pending submissions")
    
    for submission in pending:
        week = submission["week"]
        participant_id = submission["participant_id"]
        
        success = run_evaluation(week, participant_id)
        
        if success:
            update_submission_status(week, participant_id, "evaluated")
        else:
            update_submission_status(week, participant_id, "evaluation_failed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        run_once()
    else:
        watch_loop()
