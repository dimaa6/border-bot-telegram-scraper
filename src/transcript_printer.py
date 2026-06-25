import os
import sqlite3
from datetime import datetime, timezone, timedelta


def parse_latest_messages(checkpoint_id: str, dt: datetime):
    
    # Extract the chronological sliding window text from your local SQLite cache
    db_conn = sqlite3.connect(os.getenv("DB_PATH", "db/border-bot-telegram-scraper.db"))
    cursor = db_conn.cursor()

    # Let's fetch the last 30 raw text messages for the specified channel
    # Fetch the 30 most recent messages, then sort chronologically in SQL
    dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        SELECT message_id, message_text, recorded_at, reply_to_msg_id
        FROM (
            SELECT message_id, message_text, recorded_at, reply_to_msg_id
            FROM message_log
            WHERE checkpoint_id = ? AND recorded_at between datetime(?, '-8 hours') AND ?
            ORDER BY recorded_at DESC
            LIMIT 40
        )
        ORDER BY recorded_at ASC
    ''', (checkpoint_id, dt_str, dt_str))
    
    rows = cursor.fetchall()
    
    if not rows:
        db_conn.close()
        return None

    # First pass: map messages by their ID for rapid lookup    
    msg_map = {row[0]: {"text": row[1].replace('\n', ' '), "time": row[2]} for row in rows}
    
    # Second pass: Build a highly structured transcript timeline for the LLM
    transcript_lines = []
    now_utc = dt
    for msg_id, text, timestamp, reply_to_msg_id in rows:
        clean_text = text.replace('\n', ' ')
        
        try:
            msg_dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            minutes_ago = int((now_utc - msg_dt).total_seconds() / 60)
            # Ensure we don't say "-1 minutes ago" if there's a slight time sync issue
            minutes_ago = max(0, minutes_ago)
            time_label = f"{minutes_ago} minutes ago"
        except Exception:
            time_label = timestamp
            
        # Reconstruct structural context if the message is an active reply
        if reply_to_msg_id and reply_to_msg_id in msg_map:
            parent_preview = msg_map[reply_to_msg_id]["text"]
            # Truncate parent text preview to keep prompt compact but readable
            if len(parent_preview) > 60:
                parent_preview = parent_preview[:57] + "..."
            
            context_string = f"[{time_label}] ID-{msg_id} (REPLY TO ID-{reply_to_msg_id} -> '{parent_preview}'): {clean_text}"
        else:
            context_string = f"[{time_label}] ID-{msg_id}: {clean_text}"
            
        transcript_lines.append(context_string)
    
    raw_transcript = "\n".join(transcript_lines)
    return raw_transcript


print(parse_latest_messages("PL_NIZHANKOVICHI", datetime.strptime("2026-06-28 10:56:25", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)))