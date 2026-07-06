import os
import sqlite3
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

def get_chat_transcript(checkpoint_id: str, db_path: str):
    """
    Fetches the latest messages for a checkpoint, builds a structured thread-grouped
    transcript, and returns the formatted transcript string along with metadata.
    """
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()

    # Fetch the 40 most recent messages, then sort chronologically in SQL
    cursor.execute('''
        SELECT message_id, message_text, recorded_at, reply_to_msg_id
        FROM (
            SELECT message_id, message_text, recorded_at, reply_to_msg_id
            FROM message_log
            WHERE checkpoint_id = ? AND recorded_at >= datetime('now', '-8 hours')
            ORDER BY recorded_at DESC
            LIMIT 40
        )
        ORDER BY recorded_at ASC
    ''', (checkpoint_id,))
    
    rows = cursor.fetchall()
    
    if not rows:
        db_conn.close()
        return None, None, 0, None

    # First pass: map messages by their ID for rapid lookup    
    msg_map = {row[0]: {"text": row[1].replace('\n', ' '), "time": row[2]} for row in rows}
    
    # Second pass: Build a structured transcript timeline for the LLM
    now_utc = datetime.now(timezone.utc)
    
    messages = {}
    roots = []
    
    for msg_id, text, timestamp, reply_to_msg_id in rows:
        clean_text = text.replace('\n', ' ')
        
        try:
            msg_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            minutes_ago = max(0, int((now_utc - msg_dt).total_seconds() / 60))
            time_label = f"{minutes_ago} minutes ago"
        except Exception:
            time_label = timestamp
            
        messages[msg_id] = {
            "id": msg_id,
            "text": clean_text,
            "time_label": time_label,
            "reply_to": reply_to_msg_id,
            "children": [],
            "raw_timestamp": timestamp
        }

    # Build the tree of replies
    for msg_id, msg_data in messages.items():
        parent_id = msg_data["reply_to"]
        if parent_id and parent_id in messages:
            messages[parent_id]["children"].append(msg_id)
        else:
            roots.append(msg_id)
            
    transcript_lines = []
    
    def add_message_and_children(current_id):
        msg_data = messages[current_id]
        
        if msg_data["reply_to"] and msg_data["reply_to"] in messages:
            parent_id = msg_data["reply_to"]
            context_string = f"[{msg_data['time_label']}] ID-{msg_data['id']} (REPLY TO ID-{parent_id}): {msg_data['text']}"
        else:
            context_string = f"[{msg_data['time_label']}] ID-{msg_data['id']}: {msg_data['text']}"
            
        transcript_lines.append(context_string)
        
        for child_id in msg_data["children"]:
            add_message_and_children(child_id)
            
    # Iterate through roots (they are already in chronological order because dict preserves insertion order, which is from chronological query)
    for root_id in roots:
        add_message_and_children(root_id)
        
    raw_transcript = "\n".join(transcript_lines)
    
    latest_msg_dt = datetime.strptime(rows[-1][2], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    
    db_conn.close()
    
    return raw_transcript, msg_map, len(rows), latest_msg_dt


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
