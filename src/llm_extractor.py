import os
from dotenv import load_dotenv
import time
import random
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import openai
import anthropic
from typing import Literal, Any
from log_setup import configure_logging
from config_matrix import ConfigMatrix
from supabase_client import get_supabase_client, get_active_checkpoints, insert_time_stats, get_queue_history, insert_sentiment_reports
from nakordoni_client import fetch_nakordoni_data, NakordoniCheckpoint
from filter import extrapolate_trend_proxy
from transcript_builder import get_chat_transcript, cleanup_old_messages, parse_test_transcript
from llm_prompt import build_prompt, QueueReport, TimeReport, DirectionalSentiment, BorderSentimentExtraction

# Hard minimum crossing times — enforced in code after LLM response, regardless of queue size
MIN_OUTBOUND_MINUTES = 60  # Leaving Ukraine → Poland: exit control + customs + crossing + Schengen/Polish entry
MIN_INBOUND_MINUTES  = 20  # Entering Ukraine ← Poland: Polish exit + crossing + Ukrainian entry control

# --- CONFIGURATION LOADING FROM .env ---
load_dotenv()

# --- LOGGING SETUP ---
logger = configure_logging("llm_extractor.log")


FOREIGN_PATTERNS = ["польщ", "польш", "поляк"]
UKRAINE_PATTERNS = ["україн"]

FROM_PREPOSITIONS = ["з", "із", "від"]
TO_PREPOSITIONS = ["до", "в", "у", "в бік", "у бік", "в сторону", "у сторону", "на"]
EXPLICIT_INBOUND = ["додому", "до дому"]
EXPLICIT_OUTBOUND = ["на виїзд"]

def detect_direction(text: str) -> str | None:
    text = text.lower()
    text = " ".join(text.split())
    
    inbound_found = False
    outbound_found = False
    
    if any(p in text for p in EXPLICIT_INBOUND):
        inbound_found = True
    if any(p in text for p in EXPLICIT_OUTBOUND):
        outbound_found = True
        
    for f_prep in FROM_PREPOSITIONS:
        for f_pat in FOREIGN_PATTERNS:
            if f"{f_prep} {f_pat}" in text:
                inbound_found = True
        for u_pat in UKRAINE_PATTERNS:
            if f"{f_prep} {u_pat}" in text:
                outbound_found = True
                
    for t_prep in TO_PREPOSITIONS:
        for f_pat in FOREIGN_PATTERNS:
            if f"{t_prep} {f_pat}" in text:
                outbound_found = True
        for u_pat in UKRAINE_PATTERNS:
            if f"{t_prep} {u_pat}" in text:
                inbound_found = True
                
    if inbound_found and not outbound_found:
        return "INBOUND"
    elif outbound_found and not inbound_found:
        return "OUTBOUND"
    return None


def calculate_final_wait_time(base_throughput, capacity, queue_size, sentiment, floor_limit, delay=0):
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

    # Add unconditional delay (e.g. ferry embark/cross/disembark overhead) if specified
    if delay:
        final_minutes += delay

    # 3. Handle crossing overrides if available
    if sentiment.reported_crossing_minutes:
        latest_time_report = max(sentiment.reported_crossing_minutes, key=lambda x: x.source_message_id)
        final_minutes = latest_time_report.value

    # 4. Enforce floors
    return max(final_minutes, floor_limit)

def parse_latest_messages(checkpoint: dict[str, Any], llm_provider: str, ai_client, openai_client, claude_client, retry_interval: int, retry_number: int, test_transcript: str = None):
    db_path = os.getenv("DB_PATH", "db/border-bot-telegram-scraper.db")
    checkpoint_id = checkpoint["checkpoint_id"]
    
    if test_transcript:
        raw_transcript = test_transcript
        msg_map = parse_test_transcript(test_transcript)
        rows_count = len(test_transcript.splitlines())
        latest_msg_dt = datetime.now(timezone.utc)
    else:
        raw_transcript, msg_map, rows_count, latest_msg_dt = get_chat_transcript(checkpoint_id, db_path, checkpoint["lookback_hours"])

    if raw_transcript is None:
        logger.info(f"No messages cached for checkpoint {checkpoint_id}. Falling back to Nakordoni data / basic math calculation.")
        return BorderSentimentExtraction(
            from_ukraine=DirectionalSentiment(movement_state="normal"),
            to_ukraine=DirectionalSentiment(movement_state="normal")
        ), "MATH", {}, None

    prompt = f"CHAT TRANSCRIPT LOGS:\n{raw_transcript}"

    system_instruction, checkpoint_block = build_prompt(checkpoint)

    logger.info(f"Sending {rows_count} transcript lines for checkpoint {checkpoint_id} to {llm_provider}...")
    logger.debug("--- TRANSCRIPT START ---")
    logger.debug(f"\n{raw_transcript}")
    logger.debug("--- TRANSCRIPT END ---")
    
    current_wait = retry_interval
    for attempt in range(retry_number + 1):
        try:
            if llm_provider == "GEMINI":
                response = ai_client.models.generate_content(
                    model='gemini-2.5-flash',  # Fast, highly optimized for text extraction and incredibly cheap
                    contents=[checkpoint_block, prompt],
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
                    {"role": "user", "content": checkpoint_block},
                    {"role": "user", "content": prompt}
                ]
                response = openai_client.beta.chat.completions.parse(
                    model='gpt-5.6-luna',
                    messages=messages,
                    response_format=BorderSentimentExtraction,
                    # temperature=0.0,
                    reasoning_effort="medium"
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
                            "content": [
                                {"type": "text", "text": checkpoint_block, "cache_control": {"type": "ephemeral"}},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ],
                    output_config={"effort": "medium"}
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
                return BorderSentimentExtraction(
                    from_ukraine=DirectionalSentiment(movement_state="normal"),
                    to_ukraine=DirectionalSentiment(movement_state="normal")
                ), "MATH", msg_map, latest_msg_dt
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
                    if len(raw_input.keys()) == 1:  # and list(raw_input.keys())[0] in ["$PARAMETER_NAME", "query"]:
                        raw_input = list(raw_input.values())[0]
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
    logger.debug("----------------------------------------")
    logger.debug(f"RAW LLM RESPONSE:\n{raw_text}")
    logger.debug("----------------------------------------")

    # The SDK automatically handles verification and transforms the raw JSON response
    # right back into a concrete object matching your Pydantic schema structure!
    prediction_source = "LLM"
    if extracted_data is None:
        logger.error(
            f"❌ parsed data is None for {checkpoint_id} — Pydantic validation failed or model returned non-JSON.\n"
            f"Raw response text (first 1000 chars):\n{raw_text[:1000] if raw_text else '<empty>'}"
        )
        logger.warning(f"⚠️ Using fallback basic math for {checkpoint_id} due to validation failure.")
        extracted_data = BorderSentimentExtraction(
            from_ukraine=DirectionalSentiment(movement_state="normal"),
            to_ukraine=DirectionalSentiment(movement_state="normal")
        )
        prediction_source = "MATH"

    # Clean up messages older than 24 hours
    if not test_transcript:
        cleanup_old_messages(checkpoint_id, db_path)

    return extracted_data, prediction_source, msg_map, latest_msg_dt

def _build_metadata(nakordoni_cp: NakordoniCheckpoint | None, is_jammed: bool, is_warning: bool, prediction_source: str, llm_data: DirectionalSentiment | None = None, extracted_queue_size: int | None = None) -> dict:
    """Build the metadata JSONB payload stored alongside each time_stat record.

    Captures:
    - nakordoni: the raw sensor snapshot the LLM received as input (audit trail)
    - llm: the status flags the LLM derived (is_jammed, is_warning)
    - prediction_source: the source of the prediction (e.g. 'LLM', 'NAKORDONI', or 'MATH')
    """
    nakordoni_snapshot = {}
    if nakordoni_cp and nakordoni_cp.queue is not None:
        updated_at = nakordoni_cp.updated_at
        if not updated_at:
            updated_at = datetime.now(timezone.utc).isoformat()
            
        nakordoni_snapshot = {
            "queue":          nakordoni_cp.queue,
            "wait_min":       nakordoni_cp.wait_min,
            "tpercar":        None,
            "traffic_status": nakordoni_cp.traffic_status,
            "updated_at":     updated_at,
        }

    llm_snapshot = {
        "is_jammed":  is_jammed,
        "is_warning": is_warning
    }
    if llm_data:
        llm_snapshot["state"] = llm_data.movement_state or "unknown"
        llm_snapshot["queue"] = extracted_queue_size

    return {
        "nakordoni": nakordoni_snapshot,
        "llm": llm_snapshot,
        "prediction_source": prediction_source,
    }

def process_all_checkpoints():
    # Load configuration once and reuse across all checkpoint calls
    load_dotenv()

    test_transcript_path = os.getenv("TEST_TRANSCRIPT_PATH")
    test_transcript_content = None
    target_checkpoint_id = None
    if test_transcript_path:
        try:
            with open(test_transcript_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
            if lines:
                target_checkpoint_id = lines[0].strip()
                test_transcript_content = "\n".join(lines[1:])
        except Exception as e:
            logger.error(f"Failed to read test transcript from {test_transcript_path}: {e}")
            return

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

    if target_checkpoint_id:
        checkpoints = [cp for cp in checkpoints if cp["checkpoint_id"] == target_checkpoint_id]
        if not checkpoints:
            logger.error(f"Test checkpoint {target_checkpoint_id} not found in active checkpoints.")
            return
        logger.info(f"TEST MODE: Using transcript from {test_transcript_path} for checkpoint {target_checkpoint_id}")

    logger.info("Fetching official queue data from Nakordoni for all checkpoints...")
    nakordoni_all_data = fetch_nakordoni_data(supabase)
    logger.info(f"Fetched data for {len(nakordoni_all_data)} checkpoints from Nakordoni.")

    for j, cp in enumerate(checkpoints):
        checkpoint_id = cp["checkpoint_id"]

        raw_matrix = cp.get("config_matrix") or {}
        config_matrix = ConfigMatrix.model_validate(raw_matrix)

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
        metrics, prediction_source, msg_map, latest_msg_dt = parse_latest_messages(
            cp, llm_provider, ai_client, openai_client, claude_client, retry_interval, retry_number, test_transcript_content
        )
    
        if metrics:
            logger.info(f"★ SUCCESS! Type-Safe Metrics Extracted by {prediction_source} for {checkpoint_id} ★")
    
            inbound_throughput = 30
            outbound_throughput = 15
            territory_capacity = 0
            inbound_delay = 0
            outbound_delay = 0
    
            if config_matrix.ai_heuristics:
                if config_matrix.ai_heuristics.inbound_throughput:
                    inbound_throughput = config_matrix.ai_heuristics.inbound_throughput
                if config_matrix.ai_heuristics.outbound_throughput:
                    outbound_throughput = config_matrix.ai_heuristics.outbound_throughput
                if config_matrix.ai_heuristics.territory_capacity is not None:
                    territory_capacity = config_matrix.ai_heuristics.territory_capacity
                if config_matrix.ai_heuristics.inbound_delay is not None:
                    inbound_delay = config_matrix.ai_heuristics.inbound_delay
                if config_matrix.ai_heuristics.outbound_delay is not None:
                    outbound_delay = config_matrix.ai_heuristics.outbound_delay
    
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
                latest_msg_dt,
                landmark_rules,
                segment_mode,
                delay=0
            ):
                if not sentiment_data:
                    return None

                valid_qr = []
                for r in sentiment_data.reported_queue_lengths:
                    msg_info = msg_map.get(r.source_message_id)
                    if msg_info:
                        detected = detect_direction(msg_info["text"])
                        if detected and detected != direction_name:
                            logger.warning(f"Message {r.source_message_id}: assigned {direction_name} but detected {detected} — ignoring queue report")
                            continue
                    valid_qr.append(r)
                sentiment_data.reported_queue_lengths = valid_qr

                valid_tr = []
                for r in sentiment_data.reported_crossing_minutes:
                    msg_info = msg_map.get(r.source_message_id)
                    if msg_info:
                        detected = detect_direction(msg_info["text"])
                        if detected and detected != direction_name:
                            logger.warning(f"Message {r.source_message_id}: assigned {direction_name} but detected {detected} — ignoring time report")
                            continue
                    valid_tr.append(r)
                sentiment_data.reported_crossing_minutes = valid_tr

                latest_queue_report = None
                queue_size = None
                
                valid_queue_reports = [r for r in sentiment_data.reported_queue_lengths if r.value is not None or r.landmark_mentioned is not None]

                if valid_queue_reports:
                    if segment_mode == "continuous":
                        exact_reports = [r for r in valid_queue_reports if not r.is_approximate]
                        if exact_reports:
                            latest_queue_report = max(exact_reports, key=lambda x: x.source_message_id)
                        else:
                            latest_queue_report = max(valid_queue_reports, key=lambda x: x.source_message_id)
                        
                        queue_size = latest_queue_report.value if latest_queue_report else None
                    else:
                        value_reports = [r for r in valid_queue_reports if r.value is not None]
                        explicit_segment_reports = [r for r in value_reports if r.location_segment]

                        def resolve_segment(r):
                            if r.location_segment:
                                return r.location_segment.lower()
                            if explicit_segment_reports:
                                nearest = min(
                                    explicit_segment_reports,
                                    key=lambda x: abs(x.source_message_id - r.source_message_id)
                                )
                                return nearest.location_segment.lower()
                            return "barrier"

                        barrier_reports = []
                        staging_reports = []
                        for r in value_reports:
                            segment = resolve_segment(r)
                            if segment == "staging":
                                staging_reports.append(r)
                            else:
                                barrier_reports.append(r)
                        
                        latest_barrier = max(barrier_reports, key=lambda x: x.source_message_id) if barrier_reports else None
                        latest_staging = max(staging_reports, key=lambda x: x.source_message_id) if staging_reports else None

                        if latest_barrier and latest_staging:
                            barrier_msg = msg_map.get(latest_barrier.source_message_id)
                            staging_msg = msg_map.get(latest_staging.source_message_id)
                            
                            barrier_time = None
                            staging_time = None
                            if barrier_msg:
                                barrier_time = datetime.strptime(barrier_msg["time"], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                            if staging_msg:
                                staging_time = datetime.strptime(staging_msg["time"], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                                
                            if barrier_time and staging_time:
                                diff_mins = abs((barrier_time - staging_time).total_seconds()) / 60.0
                                if diff_mins > 30:
                                    if barrier_time > staging_time:
                                        queue_size = latest_barrier.value
                                    else:
                                        queue_size = latest_staging.value
                                else:
                                    queue_size = latest_barrier.value + latest_staging.value
                            else:
                                queue_size = latest_barrier.value + latest_staging.value
                        elif latest_barrier:
                            queue_size = latest_barrier.value
                        elif latest_staging:
                            queue_size = latest_staging.value

                        latest_queue_report = max(valid_queue_reports, key=lambda x: x.source_message_id)

                latest_time_report = None
                if sentiment_data.reported_crossing_minutes:
                    latest_time_report = max(sentiment_data.reported_crossing_minutes, key=lambda x: x.source_message_id)

                if latest_queue_report and latest_queue_report.landmark_mentioned and landmark_rules:
                    rules_dict = None
                    if direction_name == "INBOUND":
                        rules_dict = landmark_rules.inbound
                    elif direction_name == "OUTBOUND":
                        rules_dict = landmark_rules.outbound
                    
                    if rules_dict:
                        normalized_landmark = latest_queue_report.landmark_mentioned.lower()
                        if normalized_landmark in rules_dict:
                            landmark_queue_val = rules_dict[normalized_landmark]
                            
                            if queue_size is None:
                                queue_size = landmark_queue_val
                                logger.info(f"Resolved landmark '{normalized_landmark}' to queue size {queue_size} for {direction_name}.")
                            elif segment_mode == "continuous":
                                queue_size += landmark_queue_val
                                logger.info(f"Added landmark '{normalized_landmark}' ({landmark_queue_val}) to explicit queue size. New total: {queue_size} for {direction_name}.")

                llm_queue_size = queue_size
                direction_prediction_source = prediction_source

                if queue_size is None:
                    if nakordoni_data and nakordoni_data.queue is not None:
                        queue_size = nakordoni_data.queue
                        direction_prediction_source = "NAKORDONI"
                    else:
                        queue_size = 0
                        direction_prediction_source = "MATH"

                # if queue_size == 0:
                #     history = get_queue_history(supabase, checkpoint_id, direction_name, limit=4)
                #     if history and history[-1] >= 30:
                #         extrapolated = extrapolate_trend_proxy(history, anomaly_value=0)
                #         logger.info(f"Queue size is 0 but previous was {history[-1]}. Extrapolated to {extrapolated}.")
                #         queue_size = extrapolated

                if direction_name == "INBOUND" and queue_size < 3:
                    throughput *= 2
    
                duration = calculate_final_wait_time(
                    base_throughput=throughput,
                    capacity=territory_capacity,
                    queue_size=queue_size,
                    sentiment=sentiment_data,
                    floor_limit=floor_limit,
                    delay=delay
                )
    
                time_str = format_wait_time(duration)
                comment = f"{comment_prefix} черга у {queue_size} авто, очікування {time_str}."
    
                is_jammed = duration > jammed_threshold
                is_warning = duration > warning_threshold
    
                time_val = latest_time_report.value if latest_time_report else None
                time_src = latest_time_report.source_message_id if latest_time_report else None
                queue_val = llm_queue_size if llm_queue_size is not None else (latest_queue_report.value if latest_queue_report else None)
                queue_src = latest_queue_report.source_message_id if latest_queue_report else None

                logger.info(log_header)
                logger.info(f"Cars Queue Size:   {queue_size}")
                logger.info(f"Extracted Sentiment: {sentiment_data.movement_state}")
                logger.info(f"Extracted Time:    {time_val}, source message: {time_src}")
                logger.info(f"Extracted Queue:   {queue_val}, source message: {queue_src}")
                logger.info(f"Calculated Delay:  {duration} min")
                if delay:
                    logger.info(f"Unconditional Delay: {delay} min")
                logger.info(f"Throughput:        {throughput}")
                logger.info(f"Capacity:          {territory_capacity}")
                logger.info(f"Insight:           {comment}")
    
                extracted_at = None
                source_msg_id = time_src or queue_src
                if source_msg_id:
                    msg_info = msg_map.get(source_msg_id)
                    if msg_info:
                        extracted_at = datetime.strptime(msg_info["time"], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    
                if not extracted_at:
                    fallback_times = []
                    if latest_msg_dt:
                        fallback_times.append(latest_msg_dt)
                    if nakordoni_data:
                        if nakordoni_data.updated_at:
                            try:
                                if 'T' in nakordoni_data.updated_at:
                                    dt = datetime.fromisoformat(nakordoni_data.updated_at.replace('Z', '+00:00'))
                                else:
                                    dt = datetime.strptime(nakordoni_data.updated_at, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                                fallback_times.append(dt)
                            except Exception as e:
                                logger.warning(f"Failed to parse nakordoni timestamp: {nakordoni_data.updated_at}. Error: {e}")
                                fallback_times.append(datetime.now(timezone.utc))
                        else:
                            fallback_times.append(datetime.now(timezone.utc))
                    if fallback_times:
                        extracted_at = max(fallback_times)
    
                time_stat = {
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
                        direction_prediction_source,
                        sentiment_data,
                        llm_queue_size
                    ),
                }
    
                sentiment_payload = {
                    "checkpoint_id": checkpoint_id,
                    "direction": direction_name,
                    "transport_type": "car",
                    "movement_state": sentiment_data.movement_state,
                    "time_reports": [
                        {
                            "reported_time_minutes": r.value,
                            "source_message_id": r.source_message_id
                        } for r in sentiment_data.reported_crossing_minutes
                    ],
                    "queue_reports": [
                        {
                            "reported_queue_length": r.value,
                            "source_message_id": r.source_message_id,
                            "is_approximate": r.is_approximate,
                            "landmark_mentioned": r.landmark_mentioned,
                            "segment_mentioned": r.location_segment
                        } for r in sentiment_data.reported_queue_lengths
                    ]
                }
    
                return time_stat, sentiment_payload
    
            stats_to_insert = []
            sentiments_to_insert = []
    
            outbound_result = process_direction(
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
                latest_msg_dt=latest_msg_dt,
                landmark_rules=config_matrix.ai_heuristics.landmark_rules if config_matrix.ai_heuristics else None,
                segment_mode=config_matrix.ai_heuristics.segment_mode if config_matrix.ai_heuristics else None,
                delay=outbound_delay
            )
            if outbound_result:
                stats_to_insert.append(outbound_result[0])
                sentiments_to_insert.append(outbound_result[1])
    
            inbound_nakordoni = matched_nakordoni.get("INBOUND")
    
            inbound_result = process_direction(
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
                latest_msg_dt=latest_msg_dt,
                landmark_rules=config_matrix.ai_heuristics.landmark_rules if config_matrix.ai_heuristics else None,
                segment_mode=config_matrix.ai_heuristics.segment_mode if config_matrix.ai_heuristics else None,
                delay=inbound_delay
            )
            if inbound_result:
                stats_to_insert.append(inbound_result[0])
                sentiments_to_insert.append(inbound_result[1])
    
            if test_transcript_content:
                logger.info("TEST MODE: Skipping DB insert for file transcript execution.")
            else:
                if stats_to_insert:
                    try:
                        insert_time_stats(supabase, stats_to_insert)
                        logger.info(f"-> Saved {len(stats_to_insert)} records to 'time_stat' table in Supabase.")
                    except Exception as e:
                        logger.error(f"❌ Error saving to Supabase 'time_stat' table: {e}", exc_info=True)
    
                if sentiments_to_insert:
                    try:
                        insert_sentiment_reports(supabase, sentiments_to_insert)
                        logger.info(f"-> Saved {len(sentiments_to_insert)} sentiment records to Supabase.")
                    except Exception as e:
                        logger.error(f"❌ Error saving to Supabase sentiment tables: {e}", exc_info=True)
    
        is_last = j == (len(checkpoints) - 1)
        if not is_last:
            sleep_secs = random.uniform(45, 75) / 2.0
            logger.info(f"Sleeping {sleep_secs:.1f}s before next checkpoint (jittered, to respect API RPM limits)...")
            time.sleep(sleep_secs)

if __name__ == "__main__":
    process_all_checkpoints()
