import os
import asyncio
import sqlite3
from datetime import datetime, timezone, timedelta
from telethon import TelegramClient
from dotenv import load_dotenv
from log_setup import configure_logging
from supabase_client import get_supabase_client, get_active_checkpoints, update_supabase_state
from llm_extractor import process_all_checkpoints

# --- LOGGING SETUP ---
logger = configure_logging("scraper.log")

# --- CONFIGURATION LOADING FROM .env ---
load_dotenv()

_telegram_id_raw = os.getenv("TELEGRAM_ID")
_telegram_hash_raw = os.getenv("TELEGRAM_HASH")

if not _telegram_id_raw:
    raise EnvironmentError("Critical error: TELEGRAM_ID is not set in the .env file.")
if not _telegram_hash_raw:
    raise EnvironmentError("Critical error: TELEGRAM_HASH is not set in the .env file.")

try:
    TELEGRAM_API_ID = int(_telegram_id_raw)
except ValueError:
    raise ValueError("Critical error: TELEGRAM_ID in .env must be a valid integer.")

TELEGRAM_API_HASH = _telegram_hash_raw

# Load ignored users. Automatically convert to lowercase and strip out any accidental '@' symbols.
_ignore_raw = os.getenv("IGNORE_USERS", "")
IGNORE_USERS = [u.strip().lower().lstrip('@') for u in _ignore_raw.split(',') if u.strip()]

DB_PATH = os.getenv("DB_PATH", "db/border-bot-telegram-scraper.db")

# Initialize Supabase Admin Client
supabase = get_supabase_client()

def init_db():
    """Create the SQLite database and tables by running the external schema file."""
    SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'db_scripts', 'sqlite-schema.sql')
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema_sql)
    conn.close()
    logger.info(f"SQLite DB initialised at: {DB_PATH}")

def is_valid_message(message, ignore_users):
    """Check if the message has text and is not from an ignored user."""
    if not message.text:
        return False
    
    sender = message.sender
    if ignore_users and sender and getattr(sender, 'username', None):
        if sender.username.lower() in ignore_users:
            return False
            
    return True

def extract_message_fields(message):
    """Extract standard fields and reply_to_msg_id from a Telegram message."""
    msg_id = message.id
    sender_id = message.sender_id
    msg_date = message.date.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    text = message.text

    reply_to_id = None
    if message.reply_to and hasattr(message.reply_to, 'reply_to_msg_id'):
        reply_to_id = message.reply_to.reply_to_msg_id
        
    return msg_id, sender_id, msg_date, text, reply_to_id

def insert_message(db_cursor, cp_id, msg_id, reply_to_id, sender_id, text, msg_date):
    """Insert a message into the local database and return 1 if successful, 0 if duplicate."""
    try:
        db_cursor.execute('''
            INSERT INTO message_log (checkpoint_id, message_id, reply_to_msg_id, sender_id, message_text, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (cp_id, msg_id, reply_to_id, sender_id, text, msg_date))
        return 1
    except sqlite3.IntegrityError:
        return 0

async def scrape_active_channels():
    # 1. Fetch active targets from Supabase config
    checkpoints = get_active_checkpoints(supabase)
    
    if not checkpoints:
        logger.info("No active checkpoints to scrape.")
        return

    # 2. Connect to local SQLite cache database
    db_conn = sqlite3.connect(DB_PATH)
    db_cursor = db_conn.cursor()

    # 3. Boot up the Telegram client
    # This creates a persistent file named 'scraper.session' in your current directory
    client = TelegramClient(
        'bbhelper', 
        TELEGRAM_API_ID, 
        TELEGRAM_API_HASH,
        device_model="Web",
        system_version="WebBrowser",
        app_version="2.0.0",
        lang_code="en",
        system_lang_code="en-US"
    )
    await client.start()

    logger.info("★ Starting scraping cycle ★")

    for cp in checkpoints:
        cp_id = cp['checkpoint_id']
        handle = cp['telegram_handle']
        min_id = cp['last_message_id']

        logger.info(f"Checking {cp_id} (@{handle}) since message ID: {min_id}...")

        new_high_water_mark = min_id
        messages_inserted = 0

        try:
            lookback = cp.get('lookback_hours')
            if lookback is None:
                lookback = 3  # Fallback to 3 hours if null or not set
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback)

            # Handle cold-boot baseline safety
            if min_id is None or min_id == 0:
                logger.info(f"Cold boot detected for {cp_id}. Fetching messages up to {lookback} hours back...")
                
                first_msg = True
                # Iterate backwards in time (newest to oldest)
                async for message in client.iter_messages(handle):
                    if first_msg:
                        new_high_water_mark = message.id
                        first_msg = False
                        
                    if message.date.astimezone(timezone.utc) < cutoff_time:
                        break
                        
                    if not is_valid_message(message, IGNORE_USERS):
                        continue

                    msg_id, sender_id, msg_date, text, reply_to_id = extract_message_fields(message)

                    messages_inserted += insert_message(db_cursor, cp_id, msg_id, reply_to_id, sender_id, text, msg_date)

                db_conn.commit()
                logger.info(f"-> Local SQLite synced: Added {messages_inserted} historical updates for {cp_id}.")

                # Update Supabase instantly so the next cron run has a valid anchor
                update_supabase_state(supabase, cp_id, new_high_water_mark)
                logger.info(f"-> Cold boot complete: Baseline high_water mark pinned to {new_high_water_mark}. Inserted {messages_inserted} messages.")
                continue  # Skip to the next checkpoint in the loop

            # Regular forward scraping path (runs normally when min_id > 0)
            async for message in client.iter_messages(handle, min_id=min_id, limit=30):
                # Keep moving the marker forward unconditionally for every message seen
                if message.id > new_high_water_mark:
                    new_high_water_mark = message.id

                if message.date.astimezone(timezone.utc) < cutoff_time:
                    break

                if not is_valid_message(message, IGNORE_USERS):
                    continue

                msg_id, sender_id, msg_date, text, reply_to_id = extract_message_fields(message)

                messages_inserted += insert_message(db_cursor, cp_id, msg_id, reply_to_id, sender_id, text, msg_date)

            # Save data changes to disk
            db_conn.commit()
            logger.info(f"-> Local SQLite synced: Added {messages_inserted} new updates for {cp_id}.")

            # 4. If new logs dropped, update the Supabase state tracker for next time
            if new_high_water_mark > min_id:
                update_supabase_state(supabase, cp_id, new_high_water_mark)
                logger.info(f"-> Supabase updated: {cp_id} state high_water mark pushed to {new_high_water_mark}.")

        except Exception as e:
            logger.error(f"❌ Error scraping {handle}: {str(e)}", exc_info=True)

    # Close SQLite cleanly
    db_conn.close()
    await client.disconnect()
    logger.info("★ Cycle complete. System disconnected safely. ★")

if __name__ == "__main__":
    init_db()
    asyncio.run(scrape_active_channels())
    logger.info("★ Starting LLM Extraction Phase ★")
    process_all_checkpoints()
