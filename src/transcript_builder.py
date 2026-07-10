import os
import sqlite3
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

def get_chat_transcript(checkpoint_id: str, db_path: str, lookback_hours: int = 8):
    """
    Fetches the latest messages for a checkpoint, builds a structured thread-grouped
    transcript, and returns the formatted transcript string along with metadata.
    """
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()

    # Fetch the 40 most recent messages, then sort chronologically in SQL
    cursor.execute('''
        SELECT message_id, message_text, recorded_at, reply_to_msg_id, sender_id
        FROM (
            SELECT message_id, message_text, recorded_at, reply_to_msg_id, sender_id
            FROM message_log
            WHERE checkpoint_id = ? AND recorded_at >= datetime('now', ?)
            ORDER BY recorded_at DESC
            LIMIT 40
        )
        ORDER BY recorded_at ASC
    ''', (checkpoint_id, f'-{lookback_hours} hours'))
    
    rows = cursor.fetchall()
    
    if not rows:
        db_conn.close()
        return None, None, 0, None

    # First pass: map messages by their ID for rapid lookup    
    msg_map = {row[0]: {"text": row[1].replace('\n', ' '), "time": row[2], "sender_id": row[4]} for row in rows}
    
    # Second pass: Build a structured transcript timeline for the LLM
    now_utc = datetime.now(timezone.utc)
    
    transcript_lines = []
    
    for msg_id, text, timestamp, reply_to_msg_id, sender_id in rows:
        clean_text = text.replace('\n', ' ')
        
        try:
            msg_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            minutes_ago = max(0, int((now_utc - msg_dt).total_seconds() / 60))
            time_label = f"{minutes_ago} minutes ago"
        except Exception:
            time_label = timestamp
            
        if reply_to_msg_id:
            if reply_to_msg_id in msg_map:
                context_string = f"[{time_label}] ID-{msg_id} (REPLY TO ID-{reply_to_msg_id}): {clean_text}"
            else:
                continue
        else:
            context_string = f"[{time_label}] ID-{msg_id}: {clean_text}"  # (SENDER_ID-{sender_id})
            
        transcript_lines.append(context_string)
        
    raw_transcript = "\n".join(transcript_lines)
    
    latest_msg_dt = datetime.strptime(rows[-1][2], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    
    db_conn.close()

    return raw_transcript, msg_map, len(transcript_lines), latest_msg_dt


def cleanup_old_messages(checkpoint_id: str, db_path: str):
    """
    Cleans up messages older than 24 hours from the database.
    """
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor.execute(
            "DELETE FROM message_log WHERE checkpoint_id = ? AND recorded_at < ?",
            (checkpoint_id, cutoff_time)
        )
        db_conn.commit()
    except Exception as e:
        logger.error(f"❌ Error during local DB cleanup for {checkpoint_id}: {e}", exc_info=True)
    finally:
        db_conn.close()
