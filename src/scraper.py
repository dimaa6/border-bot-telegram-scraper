import os
import asyncio
import sqlite3
import gzip
import shutil
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone, timedelta
from telethon import TelegramClient
from supabase import create_client, Client
import tomllib  # Built-in for Python 3.11+

# --- LOGGING SETUP ---
os.makedirs("logs", exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler (Daily rotation at midnight, archives all logs indefinitely)
file_handler = TimedRotatingFileHandler(
    os.path.join("logs", "scraper.log"),
    when="midnight",
    interval=1,
    backupCount=0,
    encoding="utf-8"
)

def gzip_namer(default_name):
    return default_name + ".gz"

def gzip_rotator(source, dest):
    with open(source, 'rb') as f_in:
        with gzip.open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(source)

file_handler.namer = gzip_namer
file_handler.rotator = gzip_rotator
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("scraper")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- CONFIGURATION LOADING FROM TOML ---
CONFIG_PATH = os.path.join("config", "scraper_config.toml")

try:
    with open(CONFIG_PATH, "rb") as f:
        config_data = tomllib.load(f)

    TELEGRAM_API_ID = int(config_data["telegram_app"]["id"])
    TELEGRAM_API_HASH = str(config_data["telegram_app"]["hash"])
    SUPABASE_URL = str(config_data["supabase"]["url"])
    SUPABASE_KEY = str(config_data["supabase"]["key"])
except FileNotFoundError:
    raise FileNotFoundError(f"Critical error: Configuration file not found at '{CONFIG_PATH}'. Please verify the path.")
except KeyError as e:
    raise KeyError(f"Critical error: Missing expected key {e} inside '{CONFIG_PATH}'. Verify your TOML structure matches.")
except ValueError:
    raise ValueError(f"Critical error: 'id' in '{CONFIG_PATH}' must be a valid integer.")

# Initialize Supabase Admin Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def scrape_active_channels():
    # 1. Fetch active targets from Supabase config
    response = supabase.table("checkpoint_scraper_config") \
        .select("checkpoint_id, telegram_handle, last_message_id, lookback_hours") \
        .eq("active", True) \
        .execute()

    checkpoints = response.data
    if not checkpoints:
        logger.info("No active checkpoints to scrape.")
        return

    # 2. Connect to local SQLite cache database
    db_conn = sqlite3.connect('db/border-bot-telegram-scraper.db')
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
            # Handle cold-boot baseline safety
            if min_id is None or min_id == 0:
                lookback = cp.get('lookback_hours')
                if lookback is None:
                    lookback = 3  # Fallback to 3 hours if null or not set
                
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback)
                logger.info(f"Cold boot detected for {cp_id}. Fetching messages up to {lookback} hours back...")
                
                first_msg = True
                # Iterate backwards in time (newest to oldest)
                async for message in client.iter_messages(handle):
                    if first_msg:
                        new_high_water_mark = message.id
                        first_msg = False
                        
                    if message.date.astimezone(timezone.utc) < cutoff_time:
                        break
                        
                    if not message.text:
                        continue

                    msg_id = message.id
                    sender_id = message.sender_id
                    msg_date = message.date.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    text = message.text

                    try:
                        db_cursor.execute('''
                            INSERT INTO message_log (channel_username, message_id, sender_id, message_text, recorded_at)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (handle, msg_id, sender_id, text, msg_date))
                        messages_inserted += 1
                    except sqlite3.IntegrityError:
                        pass

                db_conn.commit()
                logger.info(f"-> Local SQLite synced: Added {messages_inserted} historical updates for {cp_id}.")

                # Update Supabase instantly so the next cron run has a valid anchor
                supabase.table("checkpoint_scraper_config") \
                    .update({
                        "last_message_id": new_high_water_mark,
                        "last_scraped_at": datetime.now(timezone.utc).isoformat()
                    }) \
                    .eq("checkpoint_id", cp_id) \
                    .execute()
                logger.info(f"-> Cold boot complete: Baseline high_water mark pinned to {new_high_water_mark}.")
                continue  # Skip to the next checkpoint in the loop

            # Regular forward scraping path (runs normally when min_id > 0)
            async for message in client.iter_messages(handle, min_id=min_id, reverse=True):
                if not message.text:
                    continue

                msg_id = message.id
                sender_id = message.sender_id
                msg_date = message.date.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                text = message.text

                try:
                    db_cursor.execute('''
                        INSERT INTO message_log (channel_username, message_id, sender_id, message_text, recorded_at)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (handle, msg_id, sender_id, text, msg_date))
                    messages_inserted += 1
                except sqlite3.IntegrityError:
                    pass

                # Keep moving the marker forward
                if msg_id > new_high_water_mark:
                    new_high_water_mark = msg_id

            # Save data changes to disk
            db_conn.commit()
            logger.info(f"-> Local SQLite synced: Added {messages_inserted} new updates for {cp_id}.")

            # 4. If new logs dropped, update the Supabase state tracker for next time
            if new_high_water_mark > min_id:
                supabase.table("checkpoint_scraper_config") \
                    .update({
                        "last_message_id": new_high_water_mark, 
                        "last_scraped_at": datetime.now(timezone.utc).isoformat()
                    }) \
                    .eq("checkpoint_id", cp_id) \
                    .execute()
                logger.info(f"-> Supabase updated: {cp_id} state high_water mark pushed to {new_high_water_mark}.")

        except Exception as e:
            logger.error(f"❌ Error scraping {handle}: {str(e)}", exc_info=True)

    # Close SQLite cleanly
    db_conn.close()
    await client.disconnect()
    logger.info("★ Cycle complete. System disconnected safely. ★")

if __name__ == "__main__":
    asyncio.run(scrape_active_channels())
