import os
import tomllib
from datetime import datetime, timezone
from supabase import create_client, Client

def get_supabase_client() -> Client:
    """Initialize and return a Supabase Admin Client."""
    CONFIG_PATH = os.path.join("config", "scraper_config.toml")
    
    with open(CONFIG_PATH, "rb") as f:
        config_data = tomllib.load(f)
        
    SUPABASE_URL = str(config_data["supabase"]["url"])
    SUPABASE_KEY = str(config_data["supabase"]["key"])
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_active_checkpoints(supabase: Client):
    """Fetch active targets from Supabase config."""
    response = supabase.table("checkpoint_scraper_config") \
        .select("checkpoint_id, telegram_handle, last_message_id, lookback_hours, config_matrix") \
        .eq("active", True) \
        .execute()
    return response.data

def update_supabase_state(supabase: Client, cp_id: str, new_high_water_mark: int):
    """Update the Supabase state tracker with the new high-water mark."""
    supabase.table("checkpoint_scraper_config") \
        .update({
            "last_message_id": new_high_water_mark,
            "last_scraped_at": datetime.now(timezone.utc).isoformat()
        }) \
        .eq("checkpoint_id", cp_id) \
        .execute()

def get_previous_estimates(supabase: Client, checkpoint_id: str) -> tuple:
    """Fetch the most recent OUTBOUND and INBOUND time_stat rows for a checkpoint.

    Returns a (prev_outbound, prev_inbound) tuple where each element is either
    a dict with keys {direction, duration_minutes, cars_queue_size, recorded_at}
    or None if no prior record exists.
    """
    rows = supabase.table("time_stat") \
        .select("direction,duration_minutes,cars_queue_size,recorded_at") \
        .eq("checkpoint_id", checkpoint_id) \
        .order("recorded_at", desc=True) \
        .limit(4) \
        .execute().data

    prev_outbound = next((r for r in rows if r["direction"] == "OUTBOUND"), None)
    prev_inbound  = next((r for r in rows if r["direction"] == "INBOUND"),  None)
    return prev_outbound, prev_inbound

def insert_time_stats(supabase: Client, stats: list) -> None:
    """Insert a list of time_stat prediction records into Supabase."""
    supabase.table("time_stat").insert(stats).execute()