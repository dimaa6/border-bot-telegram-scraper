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