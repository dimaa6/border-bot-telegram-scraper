import os
import sqlite3
from dotenv import load_dotenv
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import openai
from log_setup import configure_logging
from config_matrix import ConfigMatrix
from supabase_client import get_supabase_client, get_active_checkpoints, get_previous_estimates, insert_time_stats
from nakordoni_client import fetch_nakordoni_data, match_checkpoint_with_nakordoni, NakordoniCheckpoint

# Hard minimum crossing times — enforced in code after LLM response, regardless of queue size
MIN_OUTBOUND_MINUTES = 60  # Leaving Ukraine → Poland: exit control + customs + crossing + Schengen/Polish entry
MIN_INBOUND_MINUTES  = 20  # Entering Ukraine ← Poland: Polish exit + crossing + Ukrainian entry control

# --- LOGGING SETUP ---
logger = configure_logging("llm_extractor.log")

# 1. Define the directional schema for a single traffic flow
class DirectionalMetrics(BaseModel):
    cars_queue_size: Optional[int] = Field(
        None, description="Total passenger cars waiting outside the gates. Convert named landmarks to counts using your prompt definitions."
    )
    
    estimated_total_delay_hours: Optional[float] = Field(
        None, description="Total projected wait time in hours for a new arrival to fully cross, following the methodology steps."
    )
    
    is_jammed: bool = Field(..., description="True if total projected wait time > 3 hours or movement is at a complete standstill.")
    is_warning: bool = Field(..., description="True if there is an increasing queue, long internal wait, or notable traffic friction.")
    summary_insight: Optional[str] = Field(None, description="A concise 1-sentence summary of current conditions, written in Ukrainian, briefly explaining the key factor behind the estimate. The estimated waiting time must be formatted as 'Xгод Yхв' (e.g., 5год 40хв) with minutes rounded to the nearest 10.")

class BorderCheckpointMetrics(BaseModel):
    from_ukraine: Optional[DirectionalMetrics] = Field(
        None, description="Traffic data leaving Ukraine toward the neighboring country (e.g., 'до Польщі', 'в сторону ПЛ', 'на виїзд')."
    )
    to_ukraine: Optional[DirectionalMetrics] = Field(
        None, description="Traffic data entering Ukraine from abroad (e.g., 'додому', 'в Україну', 'на в'їзд')."
    )

def parse_latest_messages(checkpoint_id: str, config_matrix: ConfigMatrix, matched_nakordoni: dict,
                          llm_provider: str, ai_client, openai_client, retry_interval: int, retry_number: int,
                          supabase):
    
    # 4. Extract the chronological sliding window text from your local SQLite cache
    db_conn = sqlite3.connect(os.getenv("DB_PATH", "db/border-bot-telegram-scraper.db"))
    cursor = db_conn.cursor()

    # Let's fetch the last 30 raw text messages for the specified channel
    # Fetch the 30 most recent messages, then sort chronologically in SQL
    cursor.execute('''
        SELECT message_id, message_text, recorded_at, reply_to_msg_id
        FROM (
            SELECT message_id, message_text, recorded_at, reply_to_msg_id
            FROM message_log
            WHERE checkpoint_id = ?
            ORDER BY recorded_at DESC
            LIMIT 30
        )
        ORDER BY recorded_at ASC
    ''', (checkpoint_id,))
    
    rows = cursor.fetchall()
    
    if not rows:
        logger.info(f"No messages cached for checkpoint {checkpoint_id}.")
        db_conn.close()
        return None, None

    # First pass: map messages by their ID for rapid lookup    
    msg_map = {row[0]: {"text": row[1].replace('\n', ' '), "time": row[2]} for row in rows}
    
    # Second pass: Build a highly structured transcript timeline for the LLM
    transcript_lines = []
    for msg_id, text, timestamp, reply_to_msg_id in rows:
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

    # Fetch the most recent previous estimates to use as a smoothing anchor
    try:
        prev_outbound, prev_inbound = get_previous_estimates(supabase, checkpoint_id)
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch previous estimates for {checkpoint_id}: {e}")
        prev_outbound = None
        prev_inbound  = None

    def _fmt_prev(row) -> str:
        if row is None:
            return "N/A — compute freely"
        mins  = row.get('duration_minutes', 'N/A')
        queue = row.get('cars_queue_size',  'N/A')
        ts    = str(row.get('recorded_at', ''))[:16]
        return f"{mins} min wait, {queue} cars in queue (at {ts} UTC)"

    prev_outbound_str = _fmt_prev(prev_outbound)
    prev_inbound_str  = _fmt_prev(prev_inbound)

    # 5. Build a deterministic analytical prompt
    system_instruction = """You are a border crossing wait time estimator. Your task is to compute the realistic total wait time a new vehicle arrival faces right now, using sensor queue data as the primary baseline and crowdsourced chat as a source of supplementary delay information.

    PROCESSING METHODOLOGY — follow these steps in order:

    STEP 1 — BASE ESTIMATE FROM API (primary signal):
    The API queue count is sensor-based and independently verified as accurate.
    Apply the crossing time formula:
      OUTBOUND total_minutes = (queue_size + territory_capacity) / outbound_throughput * 60
      INBOUND total_minutes = (queue_size + territory_capacity) / inbound_throughput * 60
    When queue = 0 (empty-queue baseline):
      OUTBOUND total_minutes = territory_capacity / outbound_throughput * 60
      INBOUND total_minutes = territory_capacity / inbound_throughput * 60
    Set cars_queue_size to the API-reported value for that direction.

    STEP 2 — COMPLETED CROSSING CALIBRATION (highest-confidence override):
    If any chat message from the last 90 minutes reports a completed crossing
    (e.g., 'щойно пройшли за 1.5 год', 'за 2 години пройшли', 'проїхали за годину'),
    treat it as high-confidence real-world ground truth — it reflects the actual full
    experience. Adjust your estimate toward this value.

    STEP 3 — EXTRAORDINARY ANOMALY ADJUSTMENTS (additive, rare events only):
    Apply ONLY for confirmed events NOT already reflected in the queue size:
      Extra lane opened:              -15 min
      Lane reduced or closed:        +30 min
      Police / inspection activity:  +45 min
      Complete standstill (>30 min): +60 min
    Do NOT add routine internal terminal delays (e.g., 'стоїмо на території') —
    territory waiting time is already captured by territory_capacity in Step 1.
    Do NOT apply any multipliers. All adjustments are additive minutes only.

    STEP 4 — SMOOTHING ANCHOR:
    Border conditions do not change drastically within 2 hours. Your new estimate must
    stay within ±90 minutes of the previous estimate UNLESS you can cite at least 2
    independent, recent (within 90 min) chat messages confirming a significantly different
    situation. If only 1 such message exists, limit deviation to ±45 minutes.

    CROSSING TIME MODEL RULES:
    The API queue counts cars waiting OUTSIDE the gate to enter the checkpoint territory.
    The territory has a fixed capacity of cars being processed at any moment.
    When a queue exists, the territory is full. A new arrival must therefore wait for:
      (a) the external queue to drain through the gate at the corresponding throughput rate, AND
      (b) all cars already inside to finish processing before they exit.

    HARD MINIMUM FLOORS (apply even if queue is empty — non-negotiable):
      OUTBOUND (leaving Ukraine): minimum 60 minutes
        Reason: Ukrainian exit control + customs + crossing + Schengen/Polish entry.
      INBOUND  (entering Ukraine): minimum 20 minutes
        Reason: Polish exit + crossing + Ukrainian entry control.

    DIRECTIONAL ASSIGNMENT RULES:
    1. Assign data to a direction only when it carries an explicit qualifier
       ('додому'/'в Україну' = INBOUND; 'до Польщі'/'на виїзд' = OUTBOUND).
    2. Queue/lane data with NO directional qualifier → assign to OUTBOUND by default.
    3. Do NOT leave a direction null — if chat is silent for a direction, use API math from Step 1.
    4. Do NOT assign the same text segment to both directions.

    CRITICAL: Never expose internal methodology jargon to the end user. Absolute ban on phrases like "according to smoothing rules" ("згідно з правилами згладжування"),
    "due to step 4 constraints", "applying anchor limits", or "mathematical adjustments". The final text must state the situation and the final calculated time directly
    and cleanly. Explain the situation naturally and use varied phrasing. DO NOT use repetitive boilerplate templates (like always starting with "Незважаючи на чергу...").
    IMPORTANT: Format the estimated waiting time in the summary_insight as 'Xгод Yхв' (e.g., 5год 40хв), and round the number of minutes to the nearest 10. If number of
    minutes is 0, don't provide it, just say 'Xгод' (e.g., 3год).

    TONE AND STYLE INSTRUCTIONS FOR "summary_insight" (UKRAINIAN):
    1. Write naturally, as if a helpful local is updating drivers in a chat group. Avoid machine-like templates.
    2. ABSOLUTE BAN on robotic bureaucratic intros and clichés:
       - Do NOT use "Станом на зараз" (As of now).
       - Do NOT use "складає Х автомобілів" (amounts to X cars).
       - Do NOT append generic reasons at the end like "через збільшення черги" or "через невелику чергу".
    3. Do NOT just read the raw number of cars back to the user (e.g., "Черга в Україну 30 авто..."). Instead, interpret the state or describe the movement context.
    4. Vary your sentence structures. Use clean, dynamic phrasing:
       - "На виїзд доведеться почекати близько 3год 50хв."
       - "Очікуваний час проходження в Україну — орієнтовно 2год 20хв."
       - "Рух у напрямку Польщі помірний, розраховуйте приблизно на 1год 40хв."
       - "Зараз перед пунктом пропуску є накопичення, час очікування — близько 2год 30хв."
    """

    inbound_throughput = 30
    outbound_throughput = 15
    territory_capacity = 0
    landmark_rules = None

    if config_matrix.ai_heuristics:
        if config_matrix.ai_heuristics.inbound_throughput:
            inbound_throughput = config_matrix.ai_heuristics.inbound_throughput
        if config_matrix.ai_heuristics.outbound_throughput:
            outbound_throughput = config_matrix.ai_heuristics.outbound_throughput
        if config_matrix.ai_heuristics.territory_capacity is not None:
            territory_capacity = config_matrix.ai_heuristics.territory_capacity
        if config_matrix.ai_heuristics.landmark_rules:
            landmark_rules = config_matrix.ai_heuristics.landmark_rules

    landmark_section = (
        f"LOCAL LANDMARK HEURISTICS:\n    {landmark_rules}"
        if landmark_rules
        else "LOCAL LANDMARK HEURISTICS:\n    None defined — rely solely on API counts and chat mentions."
    )

    nakordoni_inbound = matched_nakordoni.get("INBOUND")
    nakordoni_outbound = matched_nakordoni.get("OUTBOUND")

    inbound_queue = f"{nakordoni_inbound.queue} cars" if nakordoni_inbound and nakordoni_inbound.queue is not None else "Unknown"
    outbound_queue = f"{nakordoni_outbound.queue} cars" if nakordoni_outbound and nakordoni_outbound.queue is not None else "Unknown"

    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    prompt = f"""
    DYNAMIC DATA MATRIX FOR THIS RUN:

    BASE DATA:
    - Current Time (UTC): {now_utc}
    - API Reported Cars in Queue (sensor-based, primary source):
      OUTBOUND (leaving Ukraine): {outbound_queue}
      INBOUND  (entering Ukraine): {inbound_queue}

    {landmark_section}

    CHECKPOINT PARAMETERS:
    Internal processing rate (throughput):
      - OUTBOUND (outbound_throughput): {outbound_throughput} cars per hour
      - INBOUND (inbound_throughput): {inbound_throughput} cars per hour
      (Rate at which the checkpoint processes cars already inside its territory)
    Territory capacity (territory_capacity): {territory_capacity} cars — physical slots inside the checkpoint
    that can be simultaneously processed.

    PREVIOUS ESTIMATES (~2 hours ago — use as smoothing anchor):
      OUTBOUND: {prev_outbound_str}
      INBOUND:  {prev_inbound_str}

    CHAT TRANSCRIPT LOGS:
    {raw_transcript}
    """


    logger.info(f"Sending {len(rows)} transcript lines for checkpoint {checkpoint_id} to {llm_provider}...")
    
    current_wait = retry_interval
    for attempt in range(retry_number + 1):
        try:
            if llm_provider == "GEMINI":
                response = ai_client.models.generate_content(
                    model='gemini-2.5-flash',  # Fast, highly optimized for text extraction and incredibly cheap
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        # These two parameters force the structured JSON extraction matching our Pydantic class
                        response_mime_type="application/json",
                        response_schema=BorderCheckpointMetrics,
                        temperature=1,  # Must be 1 when thinking_config is used (Gemini requirement)
                        thinking_config=types.ThinkingConfig(
                            # Internal chain-of-thought budget: model reasons through all 5 steps
                            # before producing the JSON. Thinking tokens are cheaper than output tokens
                            # and don't pollute the structured response schema.
                            thinking_budget=8192,
                        ),
                    ),
                )
            elif llm_provider == "GPT":
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
                response = openai_client.beta.chat.completions.parse(
                    model='gpt-5.4-mini',
                    messages=messages,
                    response_format=BorderCheckpointMetrics,
                    temperature=0.2,
                )
            else:
                raise ValueError(f"Unknown LLM Provider: {llm_provider}")
                
            break  # Success, exit the retry loop
        except Exception as e:
            err_str = str(e)
            is_retryable = '503' in err_str or '429' in err_str
            if not is_retryable or attempt == retry_number:
                logger.error(f"❌ Failed API call to {llm_provider} after {attempt} retries: {e}")
                
                logger.warning(f"⚠️ Using fallback basic math for {checkpoint_id} due to API failure.")
                prefix = checkpoint_id.split('_')[0] if '_' in checkpoint_id else ""
                country_name = {
                    "PL": "Польщі",
                    "MD": "Молдови",
                    "SK": "Словаччини",
                    "RO": "Румунії",
                    "HU": "Угорщини"
                }.get(prefix, "сусідньої країни")
                
                def format_wait_time(hours_float: float) -> str:
                    total_minutes = int(round(hours_float * 60))
                    rounded_minutes = int(round(total_minutes / 10.0) * 10)
                    h = rounded_minutes // 60
                    m = rounded_minutes % 60
                    if h > 0 and m > 0:
                        return f"{h}год {m}хв"
                    elif h > 0:
                        return f"{h}год"
                    else:
                        return f"{m}хв"

                out_queue = nakordoni_outbound.queue if nakordoni_outbound and nakordoni_outbound.queue is not None else 0
                out_hours = max(round((out_queue + territory_capacity) / outbound_throughput, 1), MIN_OUTBOUND_MINUTES / 60.0)
                out_time_str = format_wait_time(out_hours)
                
                in_queue = nakordoni_inbound.queue if nakordoni_inbound and nakordoni_inbound.queue is not None else 0
                in_hours = max(round((in_queue + territory_capacity) / inbound_throughput, 1), MIN_INBOUND_MINUTES / 60.0)
                in_time_str = format_wait_time(in_hours)
                
                from_ukr = DirectionalMetrics(
                    cars_queue_size=out_queue,
                    estimated_total_delay_hours=out_hours,
                    is_jammed=out_hours > 3.0,
                    is_warning=out_hours > 2.0,
                    summary_insight=f"На виїзд до {country_name} черга у {out_queue} авто, очікування {out_time_str}."
                )
                
                to_ukr = DirectionalMetrics(
                    cars_queue_size=in_queue,
                    estimated_total_delay_hours=in_hours,
                    is_jammed=in_hours > 2.0,
                    is_warning=in_hours > 1.5,
                    summary_insight=f"На в'їзд до України черга у {in_queue} авто, очікування {in_time_str}."
                )

                db_conn.close()
                return BorderCheckpointMetrics(from_ukraine=from_ukr, to_ukraine=to_ukr), "MATH"
            error_label = "429 Too Many Requests" if '429' in err_str else "503 Service Unavailable"
            logger.warning(f"⚠️ {error_label} — retrying in {current_wait}s ±25% (attempt {attempt + 1}/{retry_number})...")
            time.sleep(random.uniform(current_wait * 0.75, current_wait * 1.25))
            current_wait = current_wait * 2

    if llm_provider == "GEMINI":
        prompt_tokens     = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        thinking_tokens   = getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0
        cached_tokens     = getattr(response.usage_metadata, 'cached_content_token_count', 0) or 0
        total_tokens      = response.usage_metadata.total_token_count
        extracted_data    = response.parsed
        raw_text          = getattr(response, 'text', None) or str(response)
    else:
        prompt_tokens     = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
        thinking_tokens   = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0) if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens_details') else 0
        cached_tokens     = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens_details') else 0
        total_tokens      = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
        extracted_data    = response.choices[0].message.parsed
        raw_text          = response.choices[0].message.content

    logger.info("----------------------------------------")
    logger.info("📊 API TOKEN CONSUMPTION REPORT")
    logger.info("----------------------------------------")
    logger.info(f"Input Tokens  (Transcript + Prompt): {prompt_tokens}")
    logger.info(f"Cached Tokens (Prompt reuse):        {cached_tokens}")
    logger.info(f"Thinking Tokens (internal CoT):      {thinking_tokens}")
    logger.info(f"Output Tokens (Model's JSON):        {completion_tokens}")
    logger.info(f"Total Session Tokens Consumed:       {total_tokens}")
    logger.info("----------------------------------------")

    # The SDK automatically handles verification and transforms the raw JSON response
    # right back into a concrete object matching your Pydantic schema structure!
    if extracted_data is None:
        logger.error(
            f"❌ parsed data is None for {checkpoint_id} — Pydantic validation failed or model returned non-JSON.\n"
            f"Raw response text (first 1000 chars):\n{raw_text[:1000] if raw_text else '<empty>'}"
        )

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

    return extracted_data, "LLM"

def _build_metadata(nakordoni_cp: Optional[NakordoniCheckpoint], is_jammed: bool, is_warning: bool, prediction_source: str) -> dict:
    """Build the metadata JSONB payload stored alongside each time_stat record.

    Captures:
    - nakordoni: the raw sensor snapshot the LLM received as input (audit trail)
    - llm: the status flags the LLM derived (is_jammed, is_warning)
    - prediction_source: the source of the prediction (e.g. 'LLM' or 'MATH')
    """
    nakordoni_snapshot = {}
    if nakordoni_cp:
        nakordoni_snapshot = {
            "queue":          nakordoni_cp.queue,
            "wait_min":       nakordoni_cp.wait_min,
            "traffic_status": nakordoni_cp.traffic_status,
            "updated_at":     nakordoni_cp.updated_at,
        }
    return {
        "nakordoni": nakordoni_snapshot,
        "llm": {
            "is_jammed":  is_jammed,
            "is_warning": is_warning,
        },
        "prediction_source": prediction_source,
    }

def process_all_checkpoints():
    # Load configuration once and reuse across all checkpoint calls
    load_dotenv()

    llm_provider = os.getenv("LLM", "GEMINI").upper()
    
    ai_client = None
    openai_client = None
    
    if llm_provider == "GEMINI":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise EnvironmentError("Critical error: GEMINI_API_KEY is not set in the .env file.")
        ai_client = genai.Client(api_key=gemini_key)
    elif llm_provider == "GPT":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise EnvironmentError("Critical error: OPENAI_API_KEY is not set in the .env file.")
        openai_client = openai.OpenAI(api_key=openai_key)
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

    retry_interval = int(os.getenv("LLM_RETRY_INTERVAL", "30"))
    retry_number = int(os.getenv("LLM_RETRY_NUMBER", "3"))

    supabase = get_supabase_client()
    checkpoints = get_active_checkpoints(supabase)

    logger.info("Fetching official queue data from Nakordoni...")
    nakordoni_data = fetch_nakordoni_data()
    logger.info(f"Fetched data for {len(nakordoni_data)} checkpoints from Nakordoni.")

    for i, cp in enumerate(checkpoints):
        checkpoint_id = cp["checkpoint_id"]

        raw_matrix = cp.get("config_matrix") or {}
        config_matrix = ConfigMatrix(**raw_matrix)

        matched_nakordoni = match_checkpoint_with_nakordoni(config_matrix, nakordoni_data)

        logger.info(f"Processing checkpoint: {checkpoint_id}")
        metrics, prediction_source = parse_latest_messages(checkpoint_id, config_matrix, matched_nakordoni, llm_provider, ai_client, openai_client, retry_interval, retry_number, supabase)
        
        if metrics:
            logger.info(f"★ SUCCESS! Type-Safe Metrics Extracted by {prediction_source} for {checkpoint_id} ★")
            
            stats_to_insert = []

            if metrics.from_ukraine:
                outbound_raw_min = int(metrics.from_ukraine.estimated_total_delay_hours * 60) if metrics.from_ukraine.estimated_total_delay_hours is not None else MIN_OUTBOUND_MINUTES
                outbound_duration = max(outbound_raw_min, MIN_OUTBOUND_MINUTES)
                logger.info("--- FROM UKRAINE (OUTBOUND) ---")
                logger.info(f"Cars Queue Size:   {metrics.from_ukraine.cars_queue_size}")
                logger.info(f"Estimated Delay:   {metrics.from_ukraine.estimated_total_delay_hours} hours (raw) → {outbound_duration} min (floor={MIN_OUTBOUND_MINUTES}min)")
                logger.info(f"Is Jammed:         {metrics.from_ukraine.is_jammed}")
                logger.info(f"Is Warning:        {metrics.from_ukraine.is_warning}")
                logger.info(f"AI Insight:        {metrics.from_ukraine.summary_insight}")

                stats_to_insert.append({
                    "checkpoint_id": checkpoint_id,
                    "direction": "OUTBOUND",
                    "transport_type": "car",
                    "duration_minutes": outbound_duration,
                    "cars_queue_size": metrics.from_ukraine.cars_queue_size,
                    "comment": metrics.from_ukraine.summary_insight,
                    "metadata": _build_metadata(
                        matched_nakordoni.get("OUTBOUND"),
                        metrics.from_ukraine.is_jammed,
                        metrics.from_ukraine.is_warning,
                        prediction_source,
                    ),
                })

            if metrics.to_ukraine:
                inbound_raw_min = int(metrics.to_ukraine.estimated_total_delay_hours * 60) if metrics.to_ukraine.estimated_total_delay_hours is not None else MIN_INBOUND_MINUTES
                inbound_duration = max(inbound_raw_min, MIN_INBOUND_MINUTES)
                logger.info("--- TO UKRAINE (INBOUND) ---")
                logger.info(f"Cars Queue Size:   {metrics.to_ukraine.cars_queue_size}")
                logger.info(f"Estimated Delay:   {metrics.to_ukraine.estimated_total_delay_hours} hours (raw) → {inbound_duration} min (floor={MIN_INBOUND_MINUTES}min)")
                logger.info(f"Is Jammed:         {metrics.to_ukraine.is_jammed}")
                logger.info(f"Is Warning:        {metrics.to_ukraine.is_warning}")
                logger.info(f"AI Insight:        {metrics.to_ukraine.summary_insight}")

                stats_to_insert.append({
                    "checkpoint_id": checkpoint_id,
                    "direction": "INBOUND",
                    "transport_type": "car",
                    "duration_minutes": inbound_duration,
                    "cars_queue_size": metrics.to_ukraine.cars_queue_size,
                    "comment": metrics.to_ukraine.summary_insight,
                    "metadata": _build_metadata(
                        matched_nakordoni.get("INBOUND"),
                        metrics.to_ukraine.is_jammed,
                        metrics.to_ukraine.is_warning,
                        prediction_source,
                    ),
                })

            if stats_to_insert:
                try:
                    insert_time_stats(supabase, stats_to_insert)
                    logger.info(f"-> Saved {len(stats_to_insert)} records to 'time_stat' table in Supabase.")
                except Exception as e:
                    logger.error(f"❌ Error saving to Supabase 'time_stat' table: {e}", exc_info=True)

        if i < len(checkpoints) - 1:
            sleep_secs = random.uniform(45, 75)
            logger.info(f"Sleeping {sleep_secs:.1f}s before next checkpoint (jittered, to respect API RPM limits)...")
            time.sleep(sleep_secs)

if __name__ == "__main__":
    process_all_checkpoints()
