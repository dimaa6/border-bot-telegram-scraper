import os
from datetime import datetime, timezone
from supabase import create_client, Client

def get_supabase_client() -> Client:
    """Initialize and return a Supabase Admin Client."""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL:
        raise EnvironmentError("Critical error: SUPABASE_URL is not set in the .env file.")
    if not SUPABASE_KEY:
        raise EnvironmentError("Critical error: SUPABASE_KEY is not set in the .env file.")

    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_active_checkpoints(supabase: Client):
    """Fetch active targets from Supabase config."""
    response = supabase.table("checkpoint_scraper_config") \
        .select("checkpoint_id, telegram_handle, display_name, foreign_name, last_message_id, lookback_hours, config_matrix") \
        .eq("active", True) \
        .order("checkpoint_id") \
        .execute()
    return response.data

def get_active_country_prefixes(supabase: Client) -> set[str]:
    """Return the set of two-letter country prefixes (e.g. 'PL', 'HU') derived from
    the checkpoint_id of every active, non-closed checkpoint in Supabase config."""
    response = supabase.table("checkpoint_scraper_config") \
        .select("checkpoint_id") \
        .eq("active", True) \
        .eq("is_closed", False) \
        .execute()
    return {
        row["checkpoint_id"].split('_')[0]
        for row in response.data
        if row.get("checkpoint_id") and '_' in row["checkpoint_id"]
    }

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

def get_queue_history(supabase: Client, checkpoint_id: str, direction: str, limit: int = 4) -> list:
    """Fetch the last `limit` cars_queue_size values for a checkpoint and direction, returned oldest to newest."""
    rows = supabase.table("time_stat") \
        .select("cars_queue_size") \
        .eq("checkpoint_id", checkpoint_id) \
        .eq("direction", direction) \
        .order("recorded_at", desc=True) \
        .limit(limit) \
        .execute().data
    
    history = [r["cars_queue_size"] for r in rows if r.get("cars_queue_size") is not None]
    return list(reversed(history))

def insert_time_stats(supabase: Client, stats: list) -> None:
    """Insert a list of time_stat prediction records into Supabase."""
    supabase.table("time_stat").insert(stats).execute()

def insert_sentiment_reports(supabase: Client, reports: list[dict]) -> None:
    """Insert directional sentiments and their associated time and queue reports."""
    for report in reports:
        # Insert parent record
        sentiment_res = supabase.table("directional_sentiment").insert({
            "checkpoint_id": report["checkpoint_id"],
            "direction": report["direction"],
            "transport_type": report["transport_type"],
            "movement_state": report["movement_state"]
        }).execute()
        
        if sentiment_res.data:
            sentiment_id = sentiment_res.data[0]["id"]
            
            # Insert time reports
            time_reports = report.get("time_reports", [])
            if time_reports:
                time_reports_to_insert = [
                    {**tr, "directional_sentiment_id": sentiment_id}
                    for tr in time_reports
                ]
                supabase.table("time_report").insert(time_reports_to_insert).execute()
                
            # Insert queue reports
            queue_reports = report.get("queue_reports", [])
            if queue_reports:
                queue_reports_to_insert = [
                    {**qr, "directional_sentiment_id": sentiment_id}
                    for qr in queue_reports
                ]
                supabase.table("queue_report").insert(queue_reports_to_insert).execute()