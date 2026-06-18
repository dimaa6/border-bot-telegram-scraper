import os
import sqlite3
import tomllib
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from config_matrix import ConfigMatrix
from supabase_client import get_supabase_client, get_active_checkpoints
from nakordoni_client import fetch_nakordoni_data, match_checkpoint_with_nakordoni

# --- LOGGING SETUP ---
os.makedirs("logs", exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler
file_handler = logging.FileHandler(
    os.path.join("logs", "llm_extractor.log"),
    encoding="utf-8"
)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("llm_extractor")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 1. Define the directional schema for a single traffic flow
class DirectionalMetrics(BaseModel):
    # Keep the scratchpad as the first field so the model executes its logic before assigning numbers
    ai_step_by_step_analysis: str = Field(
        ..., 
        description="Mandatory logical scratchpad. Analyze and state: 1) Exterior queue size/landmark, 2) Interior terminal delays, and 3) The total math sum."
    )
    
    cars_queue_size: Optional[int] = Field(
        None, description="Total passenger cars waiting outside the gates. Convert named landmarks to counts using your prompt definitions."
    )
    
    estimated_total_delay_hours: Optional[float] = Field(
        None, description="Total projected wait time in hours for a new arrival to fully cross, matching your scratchpad calculation."
    )
    
    is_jammed: bool = Field(..., description="True if total projected wait time > 3 hours or movement is at a complete standstill.")
    is_warning: bool = Field(..., description="True if there is an increasing queue, long internal wait, or notable traffic friction.")
    summary_insight: Optional[str] = Field(None, description="A highly concise summary sentence explaining current conditions, written in Ukrainian.")

class BorderCheckpointMetrics(BaseModel):
    from_ukraine: DirectionalMetrics = Field(
        None, description="Traffic data leaving Ukraine toward the neighboring country (e.g., 'до Польщі', 'в сторону ПЛ', 'на виїзд')."
    )
    to_ukraine: DirectionalMetrics = Field(
        None, description="Traffic data entering Ukraine from abroad (e.g., 'додому', 'в Україну', 'на в'їзд')."
    )

def parse_latest_messages(checkpoint_id: str, config_matrix: ConfigMatrix, matched_nakordoni: dict):
    # 2. Load configurations from your JSON file
    CONFIG_PATH = os.path.join("config", "scraper_config.toml")
    with open(CONFIG_PATH, "rb") as f:
        config_data = tomllib.load(f)
    
    gemini_key = config_data["gemini"]["api_key"]
    retry_interval = int(config_data["gemini"].get("retry_interval", 30))
    retry_number = int(config_data["gemini"].get("retry_number", 3))
    
    # 3. Initialize the official modern GenAI SDK Client
    ai_client = genai.Client(api_key=gemini_key)
    
    # 4. Extract the chronological sliding window text from your local SQLite cache
    db_conn = sqlite3.connect('db/border-bot-telegram-scraper.db')
    cursor = db_conn.cursor()
    
    # Let's fetch the last 30 raw text messages for the specified channel
    cursor.execute('''
        SELECT message_id, message_text, recorded_at, reply_to_msg_id
        FROM message_log 
        WHERE checkpoint_id = ? 
        ORDER BY recorded_at DESC 
        LIMIT 30
    ''', (checkpoint_id,))
    
    rows = cursor.fetchall()
    
    if not rows:
        logger.info(f"No messages cached for checkpoint {checkpoint_id}.")
        db_conn.close()
        return None

    # First pass: map messages by their ID for rapid lookup    
    msg_map = {row[0]: {"text": row[1].replace('\n', ' '), "time": row[2]} for row in rows}
    
    # Second pass: Build a highly structured transcript timeline for the LLM
    transcript_lines = []
    for msg_id, text, timestamp, reply_to_msg_id in reversed(rows):
        clean_text = text.replace('\n', ' ')
        
        # Reconstruct structural context if the message is an active reply
        if reply_to_msg_id and reply_to_msg_id in msg_map:
            parent_preview = msg_map[reply_to_msg_id]["text"]
            # Truncate parent text preview to keep prompt compact but readable
            if len(parent_preview) > 60:
                parent_preview = parent_preview[:57] + "..."
            
            context_string = f"[{timestamp}] ID-{msg_id} (REPLY TO ID-{reply_to_msg_id} -> '{parent_preview}'): {clean_text}"
        else:
            context_string = f"[{timestamp}] ID-{msg_id}: {clean_text}"
            
        transcript_lines.append(context_string)
    
    raw_transcript = "\n".join(transcript_lines)    

    # 5. Build a deterministic analytical prompt
    system_instruction = "You are a border customs logistics analyzer. Your task is to calculate the absolute cumulative transit time a new vehicle arrival faces right now."

    # {profile["landmark_rules"]}
    # {profile["throughput_cars_per_hour"]}

    throughput = 15
    landmark_rules = "undefined"

    if config_matrix.ai_heuristics:
        if config_matrix.ai_heuristics.throughput:
            throughput = config_matrix.ai_heuristics.throughput
        if config_matrix.ai_heuristics.landmark_rules:
            landmark_rules = config_matrix.ai_heuristics.landmark_rules
            
    nakordoni_inbound = matched_nakordoni.get("INBOUND")
    nakordoni_outbound = matched_nakordoni.get("OUTBOUND")
    
    inbound_queue = f"{nakordoni_inbound.queue} cars" if nakordoni_inbound and nakordoni_inbound.queue is not None else "Unknown"
    outbound_queue = f"{nakordoni_outbound.queue} cars" if nakordoni_outbound and nakordoni_outbound.queue is not None else "Unknown"
    
    prompt = f"""
    You are resolving a border checkpoint wait timeline. You have been provided with two data streams:
    1. STRUCTURED BASELINE: Official/sensor metrics.
    2. UNSTRUCTURED GROUND TRUTH: Real-time user logs from crowdsourced chats.

    BASE DATA MATRIX FOR THIS RUN:
    - API Reported Cars in Queue: Inbound {inbound_queue}, Outbound {outbound_queue} vehicles

    LOCAL LANDMARK HEURISTICS:
    {landmark_rules}

    CHECKPOINT THROUGHPUT:
    Assume a baseline processing rate of {throughput} cars per hour under normal conditions.

    PROCESSING METHODOLOGY:
    1. Reconcile Vehicle Counts: Look at the 'API Reported Cars' vs. the chat mentions. If recent chat logs name a physical landmark, use the landmark heuristic count as your base 'cars_queue_size'.
    If chat logs mention exact queue size, use it. Otherwise, accept the API counts.
    2. Evaluate Internal Terminal Stagnation: Read the chat text for bottlenecks *inside* the checkpoint lines (e.g., 'стоїмо на території більше години'). The baseline API does not calculate this
    fully. You must parse these time durations explicitly.
    3. Compute Total Delay: Combine your reconciled car wait time with the parsed internal terminal bottleneck.
    4. Apply Sentiment Friction: Multiply your final projection by 1.5x if chat sentiment describes a mild slowdown, and 2.0x if describing a severe jam or standstill.
    5. Anchor Ambient Ambiguity: If a chat message describes a status, queue, or internal lane count (e.g., "на території по 7-8 машин на двох пасах") but does NOT explicitly state the direction,
    you are STRICTLY PROHIBITED from guessing or assuming it applies to the Inbound direction. If the conversation context is ambiguous, attribute it to the Outbound direction by default, or
    discard it entirely if it contradicts verified directional metrics.

    CRITICAL BIAS & ANCHORING RULES:
    1. STRICT ASSIGNMENT ONLY: Do not engage in gap-filling. If a specific traffic metric or text mention lacks a clear directional qualifier (e.g., "додому", "в UA" vs "до ПЛ", "на виїзд"), never
    guess or assume it applies to the Inbound (entering Ukraine) direction. 
    2. DIRECTIONAL ASSIGNMENT BIAS: Ground truth reports from Telegram always take priority over the official API baseline when they are clear. However, because leaving Ukraine toward Poland
    (Outbound) is structurally the much slower direction and prone to internal terminal delays, any vague, unanchored chat metrics (e.g., "7-8 cars on two lanes" or "5 cars inside") must be
    assumed to apply to the Outbound direction by default. Never assign ambiguous or directionless data to the Inbound (entering Ukraine) direction, as doing so unrealistically inflates wait
    times for what is typically a much faster flow.
    3. NO DOUBLE-COUNTING OVERLAPS: If a chat participant clarifies or updates a count (e.g., specifying that "машини на території" applies to Poland), do not split or re-assign other segments of
    that same text statement to the opposite direction without explicit textual proof.
    4. If the chat logs do not mention a direction, do NOT leave the direction null. Evaluate the 'API Reported Cars' baseline against the default processing throughput rate, and populate the metrics
    accordingly using that mathematical baseline.

    CRITICAL ALIGNMENT RULES:
    1. Every metric property in the JSON output MUST align exactly with the conclusions reached in your 'ai_step_by_step_analysis'.

    CHAT TRANSCRIPT LOGS:
    {raw_transcript}
    """

    logger.info(f"Sending {len(rows)} transcript lines for checkpoint {checkpoint_id} to Gemini 2.5 Flash...")
    
    current_wait = retry_interval
    for attempt in range(retry_number + 1):
        try:
            response = ai_client.models.generate_content(
                model='gemini-2.5-flash',  # Fast, highly optimized for text extraction and incredibly cheap
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    # These two parameters force the structured JSON extraction matching our Pydantic class
                    response_mime_type="application/json",
                    response_schema=BorderCheckpointMetrics,
                    temperature=0.1,  # Low temperature for highly deterministic analytical extractions
                ),
            )
            break  # Success, exit the retry loop
        except Exception as e:
            if '503' not in str(e) or attempt == retry_number:
                logger.error(f"❌ Failed API call to Gemini after {attempt} retries: {e}")
                raise
            logger.warning(f"⚠️ 503 Service Unavailable encountered. Retrying in {current_wait} seconds (Attempt {attempt + 1}/{retry_number})...")
            time.sleep(current_wait)
            current_wait = current_wait ** 2

    prompt_tokens = response.usage_metadata.prompt_token_count
    completion_tokens = response.usage_metadata.candidates_token_count
    total_tokens = response.usage_metadata.total_token_count

    logger.info("----------------------------------------")
    logger.info("📊 API TOKEN CONSUMPTION REPORT")
    logger.info("----------------------------------------")
    logger.info(f"Input Tokens (Transcript + Prompt): {prompt_tokens}")
    logger.info(f"Output Tokens (Gemini's JSON):     {completion_tokens}")
    logger.info(f"Total Session Tokens Consumed:     {total_tokens}")
    logger.info("----------------------------------------")

    # The SDK automatically handles verification and transforms the raw JSON response
    # right back into a concrete object matching your Pydantic schema structure!
    extracted_data = response.parsed

    # Clean up messages older than 24 hours using the existing open connection
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

    return extracted_data

def process_all_checkpoints():
    supabase = get_supabase_client()
    checkpoints = get_active_checkpoints(supabase)

    logger.info("Fetching official queue data from Nakordoni...")
    nakordoni_data = fetch_nakordoni_data()
    logger.info(f"Fetched data for {len(nakordoni_data)} checkpoints from Nakordoni.")

    # Test your logic against all active channels
    for cp in checkpoints:
        checkpoint_id = cp["checkpoint_id"]
        
        raw_matrix = cp.get("config_matrix") or {}
        config_matrix = ConfigMatrix(**raw_matrix)
        
        matched_nakordoni = match_checkpoint_with_nakordoni(config_matrix, nakordoni_data)
        
        logger.info(f"Processing checkpoint: {checkpoint_id}")
        metrics = parse_latest_messages(checkpoint_id, config_matrix, matched_nakordoni)
        
        if metrics:
            logger.info(f"★ SUCCESS! Type-Safe Metrics Extracted by Gemini for {checkpoint_id} ★")
            
            stats_to_insert = []

            if metrics.from_ukraine:
                logger.info("--- FROM UKRAINE (OUTBOUND) ---")
                logger.info(f"AI step analysis:  {metrics.from_ukraine.ai_step_by_step_analysis}")
                logger.info(f"Cars Queue Size:   {metrics.from_ukraine.cars_queue_size}")
                logger.info(f"Estimated Delay:   {metrics.from_ukraine.estimated_total_delay_hours} hours")
                logger.info(f"Is Jammed:         {metrics.from_ukraine.is_jammed}")
                logger.info(f"Is Warning:        {metrics.from_ukraine.is_warning}")
                logger.info(f"AI Insight:        {metrics.from_ukraine.summary_insight}")

                stats_to_insert.append({
                    "checkpoint_id": checkpoint_id,
                    "direction": "OUTBOUND",
                    "transport_type": "car",
                    "duration_minutes": int(metrics.from_ukraine.estimated_total_delay_hours * 60) if metrics.from_ukraine.estimated_total_delay_hours is not None else None,
                    "comment": metrics.from_ukraine.summary_insight
                })

            if metrics.to_ukraine:
                logger.info("--- TO UKRAINE (INBOUND) ---")
                logger.info(f"AI step analysis:  {metrics.to_ukraine.ai_step_by_step_analysis}")
                logger.info(f"Cars Queue Size:   {metrics.to_ukraine.cars_queue_size}")
                logger.info(f"Estimated Delay:   {metrics.to_ukraine.estimated_total_delay_hours} hours")
                logger.info(f"Is Jammed:         {metrics.to_ukraine.is_jammed}")
                logger.info(f"Is Warning:        {metrics.to_ukraine.is_warning}")
                logger.info(f"AI Insight:        {metrics.to_ukraine.summary_insight}")

                stats_to_insert.append({
                    "checkpoint_id": checkpoint_id,
                    "direction": "INBOUND",
                    "transport_type": "car",
                    "duration_minutes": int(metrics.to_ukraine.estimated_total_delay_hours * 60) if metrics.to_ukraine.estimated_total_delay_hours is not None else None,
                    "comment": metrics.to_ukraine.summary_insight
                })

            if stats_to_insert:
                try:
                    supabase.table("time_stat").insert(stats_to_insert).execute()
                    logger.info(f"-> Saved {len(stats_to_insert)} records to 'time_stat' table in Supabase.")
                except Exception as e:
                    logger.error(f"❌ Error saving to Supabase 'time_stat' table: {e}", exc_info=True)

        logger.info("Sleeping for 10 seconds to respect API RPM limits...")
        time.sleep(10)

if __name__ == "__main__":
    process_all_checkpoints()
