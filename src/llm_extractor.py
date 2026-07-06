import os
import sqlite3
from dotenv import load_dotenv
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import openai
import anthropic
from log_setup import configure_logging
from config_matrix import ConfigMatrix
from supabase_client import get_supabase_client, get_active_checkpoints, insert_time_stats, get_queue_history
from nakordoni_multi import fetch_nakordoni_multi_data, NakordoniMultiDataNode
from filter import extrapolate_trend_proxy
from transcript_builder import get_chat_transcript, cleanup_old_messages

# Hard minimum crossing times — enforced in code after LLM response, regardless of queue size
MIN_OUTBOUND_MINUTES = 60  # Leaving Ukraine → Poland: exit control + customs + crossing + Schengen/Polish entry
MIN_INBOUND_MINUTES  = 20  # Entering Ukraine ← Poland: Polish exit + crossing + Ukrainian entry control

# --- LOGGING SETUP ---
logger = configure_logging("llm_extractor.log")

class DirectionalSentiment(BaseModel):
    # Classification of overall chat traffic movement
    movement_state: Literal["normal", "slowdown", "standstill", "accelerated"] = Field(
        description="'standstill' if chat reports a complete dead stop, 'slowdown' if things are dragging/fewer lanes, 'accelerated' if moving fast, 'normal' otherwise."
    )
    # Target explicit crossing times mentioned in the chat
    reported_crossing_minutes: Optional[int] = Field(
        default=None,
        description="If a message explicitly states a completed crossing time (e.g., 'проїхали за 2 год'), extract that total time in minutes."
    )
    # Target explicit queue length mentioned in the chat
    reported_queue_length: Optional[int] = Field(
        default=None,
        description="If a message explicitly states a queue length (e.g., '25 машин'), extract that length."
    )
    # Source message id of the reported crossing time
    time_source_message_id: Optional[int] = Field(
        default=None,
        description="ID of the message you extracted the reported crossing time from."
    )
    # Source message id of the reported queue length
    queue_source_message_id: Optional[int] = Field(
        default=None,
        description="ID of the message you extracted the reported queue length from."
    )

class BorderSentimentExtraction(BaseModel):
    from_ukraine: DirectionalSentiment
    to_ukraine: DirectionalSentiment

def calculate_final_wait_time(base_throughput, capacity, queue_size, sentiment, floor_limit):
    # 1. Calculate the standard baseline minutes
    if base_throughput <= 0:
        base_throughput = 15  # Fallback to avoid division by zero
    base_hours = (queue_size + capacity) / base_throughput
    final_minutes = int(base_hours * 60)
    
    # 2. Apply layered punishments based on sentiment classification
    if sentiment.movement_state == "standstill":
        # Multiplicative: Cut throughput efficiency significantly
        final_minutes = int(final_minutes * 1.75) 
        # Additive: Tack on a flat 60-minute penalty for the dead-stop overhead
        final_minutes += 60  
        
    elif sentiment.movement_state == "slowdown":
        # Standard degradation multiplier
        final_minutes = int(final_minutes * 1.4)
        
    elif sentiment.movement_state == "accelerated":
        final_minutes = int(final_minutes * 0.75)

    # 3. Handle crossing overrides if available
    if sentiment.reported_crossing_minutes is not None:
        final_minutes = sentiment.reported_crossing_minutes

    # 4. Enforce floors
    return max(final_minutes, floor_limit)

def parse_latest_messages(checkpoint_id: str, llm_provider: str, ai_client, openai_client, claude_client, retry_interval: int, retry_number: int):
    
    db_path = os.getenv("DB_PATH", "db/border-bot-telegram-scraper.db")
    raw_transcript, msg_map, rows_count, latest_msg_dt = get_chat_transcript(checkpoint_id, db_path)
    
    if raw_transcript is None:
        logger.info(f"No messages cached for checkpoint {checkpoint_id}.")
        return None, None, None, None

    # 5. Build a deterministic analytical prompt
    system_instruction = """You are a qualitative data extraction engine for a border checkpoint. Your sole task is to analyze raw chat logs and extract traffic sentiment and direct user reports into the provided JSON schema.

INPUT STRING STRUCTURE:
Each line you analyze follows one of these two exact formats:
1. `[X minutes ago] ID-12345: Message text...` (Standalone message)
2. `[X minutes ago] ID-12345 (REPLY TO ID-67890): Message text...` (Threaded reply)

CLASSIFICATION RULES:
1. movement_state:
- "standstill": Chat reports a dead stop, complete block, or no movement for over 30 minutes.
- "slowdown": Chat complains about exceptionally slow processing, long terminal waits, or closed lanes.
- "accelerated": Chat explicitly mentions extra lanes opening or traffic clearing out rapidly.
- "normal": Default state when chat is quiet, routine, or reports steady movement.

2. reported_crossing_minutes:
- Look at the age prefix (e.g., `[45 minutes ago]`). If it is `[180 minutes ago]` or higher, you MUST ignore the line when extracting crossing times.
- If a valid message under 180 minutes reports a completed transit experience, convert the stated time directly into total minutes (e.g., "1.5 год" = 90, "3 години" = 180).

3. reported_queue_length:
- Look at the age prefix. If it is `[180 minutes ago]` or higher, you MUST ignore the line when extracting queue lengths.
- If a valid message explicitly states a queue length (e.g., '25 машин', '3 авто'), extract that integer.
- If a message says "відразу на територію", "відразу на кордон", "пусто", or "немає черги", the queue length is 0.

4. time_source_message_id & queue_source_message_id:
- You MUST populate these fields with the exact numeric digits following the `ID-` tag of the primary line (not the reply-to ID). For example, in `ID-285929`, the ID is 285929. If no data is extracted, leave as null.

CRITICAL TRANSCRIPT FILTERING RULES:
- Ignore all questions, requests for updates, and info-seeking messages (e.g., messages containing "Підкажіть", "яка черга?", "чи є рух?", "хто знає"). They do NOT represent real conditions.
- Only extract state classifications from factual assertions or direct driver updates (e.g., "пусто", "стоїмо", "проїхали за...").

FOREIGN CHECKPOINT FIREWALL:
- Drivers frequently reference entirely different borders in chat threads. If the main message text or the parent message preview explicitly mentions a different checkpoint by name (e.g., "Угринів", "Uhryniv", "Ягодин", "Краківець", "Рава", "Грушів", "Шегині"), you MUST completely ignore that line. NEVER extract queue sizes, wait times, or sentiment from lines mentioning a foreign checkpoint.

CRITICAL DIRECTIONAL OVERRIDE RULES (PASSENGER CARS ONLY):
1. 'to_ukraine' (Entering Ukraine, Inbound) STRICT DEFINITION:
   - A message belongs to this direction ONLY if it contains phrases explicitly showing movement TOWARDS Ukraine.
   - Key tokens: "в Україну", "до України", "в сторону України", "на в'їзд", "додому", "на UA".

2. 'from_ukraine' (Leaving Ukraine / Going abroad, Outbound) STRICT DEFINITION:
   - A message belongs to this direction ONLY if it contains phrases showing movement AWAY from Ukraine / TOWARDS Poland.
   - Key tokens: "до Польщі", "в Польщу", "в сторону Польщі", "на виїзд", "на ПЛ", "на виїзд з UA".

3. STRICT THREAD LINKAGE & CONTEXT INHERITANCE:
   - If a line is a threaded reply containing `(REPLY TO ID-XXXX)`:
     * You MUST look at the preceding lines in the transcript to find the message that matches `ID-XXXX`.
     * Inherit the directional context strictly from that parent message. If the parent message asked "Яка черга до України?", then this reply is strictly linked to 'to_ukraine'.
   - If a line has NO explicit direction tokens AND has no `(REPLY TO ID-XXXX)` metadata block, you are strictly FORBIDDEN from guessing. Leave both directions as null.

4. THE BOUNDARY WALL (WHEN TO WRITE NULL):
   - If the text or inherited parent context contains "України" or "Україна" preceded by "в", "до", or "в сторону", it is mathematically IMPOSSIBLE for it to be outbound. You are strictly forbidden from placing this data in 'from_ukraine'.
   - If there is even a 1% ambiguity or lack of clear directional tokens in both the text and the parent string block, output null for all fields. Never assume.

ABSOLUTE NUMERICAL FILTERING:
- NEVER extract numbers or sentiment from questions, emotional outcries, or messages mocking/repeating a previous statement (e.g., "Яких 0?", "Звідки там 5 годин?!"). 
- Only extract queue sizes if they are part of a direct, affirmative factual update from a driver.
"""

# """FEW-SHOT TRAINING EXAMPLES FOR COLD START ALIGNMENT:

# EXAMPLE 1 ANALYSIS TASK:
# Transcript:
# [2026-06-25 10:53] ID-101: А в Україну є черга?
# [2026-06-25 10:56] ID-102: Додому пусто
# [2026-06-25 11:32] ID-103: Вітаю! Підкажіть будь ласка, на цей момент в Україну яка черга?
# Target Output JSON Representation:
# {
#   "from_ukraine": { "movement_state": "normal", "reported_crossing_minutes": null },
#   "to_ukraine": { "movement_state": "normal", "reported_crossing_minutes": null }
# }

# EXAMPLE 2 ANALYSIS TASK:
# Transcript:
# [2026-06-25 12:10] ID-201: Черга жах, стоїмо на одному місці вже годину, жодна машина не проїхала
# [2026-06-25 12:15] ID-202: До Польщі повний стоп, митники не працюють взагалі
# Target Output JSON Representation:
# {
#   "from_ukraine": { "movement_state": "standstill", "reported_crossing_minutes": null },
#   "to_ukraine": { "movement_state": "normal", "reported_crossing_minutes": null }
# }

# EXAMPLE 3 ANALYSIS TASK:
# Transcript:
# [2026-06-25 14:05] ID-301: Привіт усім. Проїхали до Польщі щойно. Загалом все зайняло десь 2.5 години від сили. Рух є.
# [2026-06-25 14:12] ID-302: Хто знає як автобуси йдуть на вїзд?
# Target Output JSON Representation:
# {
#   "from_ukraine": { "movement_state": "normal", "reported_crossing_minutes": 150 },
#   "to_ukraine": { "movement_state": "normal", "reported_crossing_minutes": null }
# }

# EXAMPLE 4 ANALYSIS TASK:
# Transcript:
# [2026-06-25 15:20] ID-401: На території капець, оформляють дуже повільно, працює всього один пас. Навіть перед шлагбаумом черга росте.
# Target Output JSON Representation:
# {
#   "from_ukraine": { "movement_state": "slowdown", "reported_crossing_minutes": null },
#   "to_ukraine": { "movement_state": "normal", "reported_crossing_minutes": null }
# }
# """

    prompt = f"CHAT TRANSCRIPT LOGS:\n{raw_transcript}"

    logger.info(f"Sending {rows_count} transcript lines for checkpoint {checkpoint_id} to {llm_provider}...")
    
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
                        response_schema=BorderSentimentExtraction,
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
                    response_format=BorderSentimentExtraction,
                    temperature=0.0,
                )
            elif llm_provider == "CLAUDE":
                schema = BorderSentimentExtraction.model_json_schema()
                response = claude_client.messages.create(
                    model='claude-sonnet-5',
                    max_tokens=1024,
                    system=[
                        {
                            "type": "text",
                            "text": system_instruction,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    tools=[
                        {
                            "name": "extract_sentiment",
                            "description": "Extract sentiment from border traffic chat transcript.",
                            "input_schema": schema
                        }
                    ],
                    tool_choice={"type": "tool", "name": "extract_sentiment"},
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
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
                
                db_conn.close()
                return BorderSentimentExtraction(
                    from_ukraine=DirectionalSentiment(movement_state="normal", reported_crossing_minutes=None),
                    to_ukraine=DirectionalSentiment(movement_state="normal", reported_crossing_minutes=None)
                ), "MATH"
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
    elif llm_provider == "CLAUDE":
        prompt_tokens     = getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0
        completion_tokens = getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0
        thinking_tokens   = getattr(response.usage, 'thinking_tokens', 0) if hasattr(response, 'usage') else 0
        cached_tokens     = getattr(response.usage, 'cache_read_input_tokens', 0) if hasattr(response, 'usage') else 0
        total_tokens      = prompt_tokens + completion_tokens
        
        extracted_data = None
        for block in getattr(response, 'content', []):
            if getattr(block, 'type', '') == "tool_use" and getattr(block, 'name', '') == "extract_sentiment":
                try:
                    raw_input = getattr(block, 'input', {})
                    extracted_data = BorderSentimentExtraction.model_validate(raw_input)
                except Exception as e:
                    logger.error(f"Failed to parse Claude output: {e} | Raw Input: {raw_input}")
                break
        raw_text = str(getattr(response, 'content', ''))
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

    # Clean up messages older than 24 hours
    cleanup_old_messages(checkpoint_id, db_path)

    return extracted_data, "LLM", msg_map, latest_msg_dt

def _build_metadata(nakordoni_cp: Optional[NakordoniMultiDataNode], is_jammed: bool, is_warning: bool, prediction_source: str, llm_data: Optional[DirectionalSentiment] = None) -> dict:
    """Build the metadata JSONB payload stored alongside each time_stat record.

    Captures:
    - nakordoni: the raw sensor snapshot the LLM received as input (audit trail)
    - llm: the status flags the LLM derived (is_jammed, is_warning)
    - prediction_source: the source of the prediction (e.g. 'LLM' or 'MATH')
    """
    nakordoni_snapshot = {}
    if nakordoni_cp and nakordoni_cp.queue:
        updated_at = None
        if nakordoni_cp.update_info and nakordoni_cp.update_info.timestamp:
            dt = datetime.fromtimestamp(nakordoni_cp.update_info.timestamp, timezone.utc)
            updated_at = dt.isoformat()
            
        nakordoni_snapshot = {
            "queue":          nakordoni_cp.queue.queue_now,
            "wait_min":       nakordoni_cp.queue.wait_min,
            "tpercar":        nakordoni_cp.queue.tpercar,
            "traffic_status": None,
            "updated_at":     updated_at,
        }

        llm_snapshot = {
            "is_jammed":  is_jammed,
            "is_warning": is_warning
        }
        if llm_data:
            llm_snapshot["state"] = llm_data.movement_state or "unknown"
            llm_snapshot["queue"] = llm_data.reported_queue_length or "unknown"

    return {
        "nakordoni": nakordoni_snapshot,
        "llm": llm_snapshot,
        "prediction_source": prediction_source,
    }

def process_all_checkpoints():
    # Load configuration once and reuse across all checkpoint calls
    load_dotenv()

    llm_provider = os.getenv("LLM", "GEMINI").upper()
    
    ai_client = None
    openai_client = None
    claude_client = None
    
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
    elif llm_provider == "CLAUDE":
        claude_key = os.getenv("CLAUDE_API_KEY")
        if not claude_key:
            raise EnvironmentError("Critical error: CLAUDE_API_KEY is not set in the .env file.")
        claude_client = anthropic.Anthropic(api_key=claude_key)
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

    retry_interval = int(os.getenv("LLM_RETRY_INTERVAL", "30"))
    retry_number = int(os.getenv("LLM_RETRY_NUMBER", "3"))

    supabase = get_supabase_client()
    checkpoints = get_active_checkpoints(supabase)

    chunk_size = 10
    nakordoni_all_data = {}
    
    for chunk_start in range(0, len(checkpoints), chunk_size):
        chunk_checkpoints = checkpoints[chunk_start:chunk_start + chunk_size]
        
        ppids_to_fetch = []
        for cp in chunk_checkpoints:
            raw_matrix = cp.get("config_matrix") or {}
            config_matrix = ConfigMatrix(**raw_matrix)
            if config_matrix.nakordoni and config_matrix.nakordoni.car:
                if config_matrix.nakordoni.car.inbound_id:
                    ppids_to_fetch.append(config_matrix.nakordoni.car.inbound_id)
                if config_matrix.nakordoni.car.outbound_id:
                    ppids_to_fetch.append(config_matrix.nakordoni.car.outbound_id)
                    
        ppids_to_fetch = list(set(ppids_to_fetch))
        
        if ppids_to_fetch:
            logger.info(f"Fetching official queue data from Nakordoni for {len(ppids_to_fetch)} ppids...")
            chunk_data = fetch_nakordoni_multi_data(ppids_to_fetch)
            nakordoni_all_data.update(chunk_data)
            logger.info(f"Fetched data for {len(chunk_data)} checkpoints from Nakordoni.")

    for j, cp in enumerate(checkpoints):
        checkpoint_id = cp["checkpoint_id"]

        raw_matrix = cp.get("config_matrix") or {}
        config_matrix = ConfigMatrix(**raw_matrix)

        matched_nakordoni = {
            "INBOUND": None,
            "OUTBOUND": None
        }
        if config_matrix.nakordoni and config_matrix.nakordoni.car:
            if config_matrix.nakordoni.car.inbound_id:
                matched_nakordoni["INBOUND"] = nakordoni_all_data.get(config_matrix.nakordoni.car.inbound_id)
            if config_matrix.nakordoni.car.outbound_id:
                matched_nakordoni["OUTBOUND"] = nakordoni_all_data.get(config_matrix.nakordoni.car.outbound_id)

        logger.info(f"Processing checkpoint: {checkpoint_id}")
        metrics, prediction_source, msg_map, latest_msg_dt = parse_latest_messages(checkpoint_id, llm_provider, ai_client, openai_client, claude_client, retry_interval, retry_number)
    
        if metrics:
            logger.info(f"★ SUCCESS! Type-Safe Metrics Extracted by {prediction_source} for {checkpoint_id} ★")
    
            inbound_throughput = 30
            outbound_throughput = 15
            territory_capacity = 0
    
            if config_matrix.ai_heuristics:
                if config_matrix.ai_heuristics.inbound_throughput:
                    inbound_throughput = config_matrix.ai_heuristics.inbound_throughput
                if config_matrix.ai_heuristics.outbound_throughput:
                    outbound_throughput = config_matrix.ai_heuristics.outbound_throughput
                if config_matrix.ai_heuristics.territory_capacity is not None:
                    territory_capacity = config_matrix.ai_heuristics.territory_capacity
    
            prefix = checkpoint_id.split('_')[0] if '_' in checkpoint_id else ""
            country_name = {
                "PL": "Польщі",
                "MD": "Молдови",
                "SK": "Словаччини",
                "RO": "Румунії",
                "HU": "Угорщини"
            }.get(prefix, "сусідньої країни")
    
            def format_wait_time(total_minutes: int) -> str:
                rounded_minutes = int(round(total_minutes / 10.0) * 10)
                h = rounded_minutes // 60
                m = rounded_minutes % 60
                if h > 0 and m > 0:
                    return f"{h}год {m}хв"
                elif h > 0:
                    return f"{h}год"
                else:
                    return f"{m}хв"
    
            def process_direction(
                sentiment_data, 
                direction_name, 
                nakordoni_data, 
                throughput, 
                floor_limit, 
                jammed_threshold, 
                warning_threshold, 
                comment_prefix,
                log_header,
                msg_map,
                latest_msg_dt
            ):
                if not sentiment_data:
                    return None

                queue_size = sentiment_data.reported_queue_length

                if queue_size is None:
                    queue_size = nakordoni_data.queue.queue_now if nakordoni_data and nakordoni_data.queue and nakordoni_data.queue.queue_now is not None else 0

                if queue_size == 0:
                    history = get_queue_history(supabase, checkpoint_id, direction_name, limit=4)
                    if history and history[-1] >= 30:
                        extrapolated = extrapolate_trend_proxy(history, anomaly_value=0)
                        logger.info(f"Queue size is 0 but previous was {history[-1]}. Extrapolated to {extrapolated}.")
                        queue_size = extrapolated

                if direction_name == "INBOUND" and queue_size < 3:
                    throughput *= 2
    
                duration = calculate_final_wait_time(
                    base_throughput=throughput,
                    capacity=territory_capacity,
                    queue_size=queue_size,
                    sentiment=sentiment_data,
                    floor_limit=floor_limit
                )
    
                time_str = format_wait_time(duration)
                comment = f"{comment_prefix} черга у {queue_size} авто, очікування {time_str}."
    
                is_jammed = duration > jammed_threshold
                is_warning = duration > warning_threshold
    
                logger.info(log_header)
                logger.info(f"Cars Queue Size:   {queue_size}")
                logger.info(f"Extracted Sentiment: {sentiment_data.movement_state}")
                logger.info(f"Extracted Time:    {sentiment_data.reported_crossing_minutes}, source message: {sentiment_data.time_source_message_id}")
                logger.info(f"Extracted Queue:   {sentiment_data.reported_queue_length}, source message: {sentiment_data.queue_source_message_id}")
                logger.info(f"Calculated Delay:  {duration} min")
                logger.info(f"Throughput:        {throughput}")
                logger.info(f"Capacity:          {territory_capacity}")
                logger.info(f"Insight:           {comment}")
    
                extracted_at = None
                if sentiment_data and sentiment_data.time_source_message_id:
                    msg_info = msg_map.get(sentiment_data.time_source_message_id)
                    if msg_info:
                        extracted_at = datetime.strptime(msg_info["time"], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    
                if not extracted_at:
                    fallback_times = []
                    if latest_msg_dt:
                        fallback_times.append(latest_msg_dt)
                    if nakordoni_data and nakordoni_data.update_info and nakordoni_data.update_info.timestamp:
                        try:
                            dt = datetime.fromtimestamp(nakordoni_data.update_info.timestamp, timezone.utc)
                            fallback_times.append(dt)
                        except Exception as e:
                            logger.warning(f"Failed to parse nakordoni timestamp: {nakordoni_data.update_info.timestamp}. Error: {e}")
                    if fallback_times:
                        extracted_at = max(fallback_times)
    
                return {
                    "checkpoint_id": checkpoint_id,
                    "direction": direction_name,
                    "transport_type": "car",
                    "duration_minutes": duration,
                    "cars_queue_size": queue_size,
                    "comment": comment,
                    "extracted_at": extracted_at.isoformat() if extracted_at else None,
                    "metadata": _build_metadata(
                        nakordoni_data,
                        is_jammed,
                        is_warning,
                        prediction_source,
                        sentiment_data
                    ),
                }
    
            stats_to_insert = []
    
            outbound_stat = process_direction(
                sentiment_data=metrics.from_ukraine,
                direction_name="OUTBOUND",
                nakordoni_data=matched_nakordoni.get("OUTBOUND"),
                throughput=outbound_throughput,
                floor_limit=MIN_OUTBOUND_MINUTES,
                jammed_threshold=270,
                warning_threshold=180,
                comment_prefix=f"На виїзд до {country_name}",
                log_header="--- FROM UKRAINE (OUTBOUND) ---",
                msg_map=msg_map,
                latest_msg_dt=latest_msg_dt
            )
            if outbound_stat:
                stats_to_insert.append(outbound_stat)
    
            inbound_nakordoni = matched_nakordoni.get("INBOUND")
    
            inbound_stat = process_direction(
                sentiment_data=metrics.to_ukraine,
                direction_name="INBOUND",
                nakordoni_data=inbound_nakordoni,
                throughput=inbound_throughput,
                floor_limit=MIN_INBOUND_MINUTES,
                jammed_threshold=120,
                warning_threshold=90,
                comment_prefix="На в'їзд до України",
                log_header="--- TO UKRAINE (INBOUND) ---",
                msg_map=msg_map,
                latest_msg_dt=latest_msg_dt
            )
            if inbound_stat:
                stats_to_insert.append(inbound_stat)
    
            if stats_to_insert:
                try:
                    insert_time_stats(supabase, stats_to_insert)
                    logger.info(f"-> Saved {len(stats_to_insert)} records to 'time_stat' table in Supabase.")
                except Exception as e:
                    logger.error(f"❌ Error saving to Supabase 'time_stat' table: {e}", exc_info=True)
    
        is_last = j == (len(checkpoints) - 1)
        if not is_last:
            sleep_secs = random.uniform(45, 75)
            logger.info(f"Sleeping {sleep_secs:.1f}s before next checkpoint (jittered, to respect API RPM limits)...")
            time.sleep(sleep_secs)

if __name__ == "__main__":
    process_all_checkpoints()
