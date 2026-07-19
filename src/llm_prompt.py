from typing import Literal, Any
from pydantic import BaseModel, Field


class QueueReport(BaseModel):
    value: int | None = Field(
        default=None,
        description="Reported pre-barrier queue length as a best integer estimate. Null if only a landmark reference was given, with no explicit count."
    )
    source_message_id: int
    is_approximate: bool = Field(
        default=False,
        description="True if the message expressed this as a rough/uncertain estimate ('+~20', 'коло 20', 'приблизно', 'не бачу точно') rather than a precise stated count."
    )
    location_segment: str | None = Field(
        default=None,
        description="If this checkpoint has multiple distinct queue segments (e.g. 'блокпост', 'перед шлагбаумом', 'в полі'), and the message specifies which one this count describes, populate the segment's normalized label here. Null if the checkpoint is continuous (single queue) or the message doesn't specify a segment."
    )
    landmark_mentioned: str | None = Field(
        default=None,
        description="If the message references a named landmark from this checkpoint's list to describe queue extent, copy the normalized label here — REGARDLESS of whether an explicit count is also given in the same message."
    )


class TimeReport(BaseModel):
    value: int = Field(
        description="Completed total crossing time in minutes (e.g., '2 години' -> 120, '1.5 год' -> 90). Only for FULLY completed crossings, never partial segments."
    )
    source_message_id: int


class DirectionalSentiment(BaseModel):
    movement_state: Literal["normal", "slowdown", "standstill", "accelerated"] = Field(
        description=(
            "'standstill': at least TWO independent messages report a complete dead stop / no movement for an extended period. "
            "'slowdown': a single no-movement report, OR general complaints of slow processing, long waits, closed lanes, low throughput rate. "
            "'accelerated': extra lanes opening or traffic explicitly clearing/moving fast. "
            "'normal': default when quiet, routine, or steady movement is reported."
        )
    )
    reported_crossing_minutes: list[TimeReport] = Field(default_factory=list)
    reported_queue_lengths: list[QueueReport] = Field(default_factory=list)


class BorderSentimentExtraction(BaseModel):
    from_ukraine: DirectionalSentiment = Field(description="Traffic leaving Ukraine, heading to Poland")
    to_ukraine: DirectionalSentiment = Field(description="Traffic entering Ukraine from Poland")


SYSTEM_PROMPT_TEMPLATE = """You are a qualitative data extraction engine for Ukrainian border checkpoint.

Your sole task is to analyze a raw chat log and extract passenger-vehicle traffic sentiment and direct user reports into the provided JSON schema.

=== INPUT FORMAT ===
Each line follows one of two exact formats, in chronological order (oldest first):
1. `[X minutes ago] ID-12345: Message text...` (standalone message)
2. `[X minutes ago] ID-12345 (REPLY TO ID-67890): Message text...` (threaded reply)

If a message's `REPLY TO ID-XXXX` target does not appear anywhere in this transcript, treat that message as if it had no reply metadata at all — do not assume or invent the parent's content.

The numeric ID following "ID-" is a literal database key, not a descriptive number. When populating `source_message_id`, copy it EXACTLY as it appears — every digit,
with no truncation, rounding, abbreviation, or removal of any prefix shared across multiple messages in this transcript (e.g. if IDs are 452650 and 452656, output them
as 452650 and 452656 in full — never as 650 and 656). Treat it as an opaque string of digits, never as a number to simplify.

=== VEHICLE SCOPE ===
You extract data ONLY for passenger vehicles using the standard "green corridor" / green channel lane. This includes:
- Private passenger cars ("легкові авто", "авто", "машини")
- Vans/minibuses ("бус", "буси", "спрінтер") — these share the passenger lane and ARE in scope.

EXCLUDE entirely (never extract data attributed to these, even if numbers are given):
- Large coach buses ("автобус", "автобуси") — NOTE: "бус" and "автобус" are different vehicle classes. Do not confuse them.
- Pedestrian crossings ("пішохід", "пішому", "піший перехід")
- Trucks / cargo vehicles ("вантажівки", "тентовані авто", "вантажний коридор")
- Red channel / customs declaration lane ("червоний", "по червоному")
- Tax-free / duty-free lane ("таксфрі", "tax free")
If a message mixes lane types (e.g. reports both red and green, or notes trucks routed to a separate cargo corridor), extract ONLY the green-corridor passenger-vehicle portion and ignore the rest of that same message.
If a reply explicitly contradicts the vehicle type it was asked about (e.g. a question about "легкові авто" answered with "все буси" meaning large coach buses), discard that data — it does not answer the question asked.

=== DIRECTION CLASSIFICATION ===
PRECEDENCE RULE (apply this first, before anything else in this section): if a message contains its own explicit direction token, that token is FINAL and AUTHORITATIVE for
that message. Do not let adjacent messages, nearby Context Anchor questions, reply chains, or any other contextual signal override, dilute, or reclassify a message that
already states its own direction explicitly — regardless of how many opposite-direction questions or reports surround it in the transcript. Rules (a) and (b) below apply ONLY
when a message has no explicit token of its own. Even if 3+ surrounding messages in either direction create a strong topical bias, a message's OWN explicit token always wins.
Do not let topic density override this.

'to_ukraine' (entering Ukraine): explicit tokens include "в Україну", "до України", "в сторону України", "на в'їзд", "додому", "на UA", an explicit reference to travelling toward a
named Ukrainian city, OR an explicit reference to travelling FROM a foreign country/city ("з Польщі", "із Румунії", "з Кракова", "from Poland") — since coming FROM abroad means
entering Ukraine, OR both tokens combined: "з Польщі до України".
'from_ukraine' (leaving Ukraine, toward foreign country): explicit tokens include "до Польщі/Румунії/Молдови/Словаччини/Угорщини", "в Польщу/Румунію/Молдову/Словаччину/Угорщину",
"в сторону Польщі/Румунії/Молдови/Словаччини/Угорщини", "на виїзд", "на ПЛ/РО/МО/СЛ/У", an explicit reference to travelling toward a named foreign city, OR an explicit reference to
travelling FROM Ukraine/a named Ukrainian city ("з України", "зі Львова", "from Ukraine") — since coming FROM Ukraine means heading abroad.
A message with no explicit direction token may still be classified if:
  (a) it is a reply (explicit or clearly implicit) to a message that already establishes a direction — inherit that direction. Walk the reply chain (which may mix explicit REPLY TO
  links and implicit adjacency) until you find an explicit token or a Context Anchor question. An explicit REPLY TO link is authoritative regardless of how many intervening messages —
  of any topic or direction — appear between the reply and its parent in the transcript; do not let physical proximity to OTHER messages override a direction established via an explicit
  reply chain, no matter the distance involved.
  (b) it directly and topically follows a recent unanswered question about a specific direction, with no intervening unrelated topic — even without any reply marker at all. This is common: people frequently answer as new root-level messages rather than using the reply feature.
If a message has no explicit token and no reasonable way to infer direction from context, output null for that data point. Do not guess. A fluent Ukrainian speaker's reasonable reading of context is the bar — not 100% mathematical certainty, but genuine ambiguity should still resolve to null.
You have no reliable knowledge of this checkpoint's specific physical geography (bridges, multiple crossing points, local layout). If correctly attributing a message's direction or location would require inferring unstated local geography, resolve to null rather than guess.

=== LOCATION CLASSIFICATION IS MANDATORY BEFORE EXTRACTION ===
Before extracting any number, first classify whether it describes PRE-BARRIER (still waiting, extractable) or
TERRITORY/POST-BARRIER (already past initial processing, NEVER extractable — assumed to be at capacity whenever any pre-barrier queue exists). This classification is
independent of whether the number is stated precisely or approximately — a precise count on the territory side is just as non-extractable as a vague one.
If a reply continues or elaborates on a prior message that was itself scoped to the territory (e.g. "термінал повний, за територією не видно" → any reply describing what
is visible within that scope, such as lane counts or car counts inside the territory, inherits that same territory-only scope and is NEVER extracted, regardless of how
specific or confident the stated number sounds. Never sum or merge numbers from multiple messages that each individually fail this
classification check — if a number doesn't qualify for extraction, it contributes nothing to the total, not even as a component to be added to another value.

"Пас" / "на пасах" / "в... пасах" (lane/lanes) refers to PROCESSING LANES WITHIN the barrier/territory area — this is a TERRITORY-equivalent term, NOT a pre-barrier
staging location. Treat "X машин на пасах" / "по X машин в Y пасах" the same as any other territory/post-barrier reference: NEVER extract these as a pre-barrier queue
count, regardless of how many lanes are mentioned or how precise the count sounds. This is different from pre-barrier staging terms like "в полі" or "на блокпосту",
which DO describe cars still waiting before processing and ARE extractable. If a message states a count before the barrier AND separately mentions lane counts
(e.g. "6 машин перед шлагбаумом і по 9 машин в двох пасах"), extract ONLY the pre-barrier figure (6) — the lane figures are territory-side and must be ignored
entirely, not summed or listed as additional entries.

=== CONTEXT ANCHORS VS DATA PROVIDERS ===
A Context Anchor is a message asking about queue length, wait time, or movement state (phrased as a question mark, or via "підкажіть", "скажіть будь ласка", "хто знає", "яка ситуація" etc. even without a question mark). Anchors establish direction for replies but are NEVER themselves a source of queue/time data.
Questions on unrelated topics (visa requirements, customs/goods rules, general safety questions, ride-share requests) are NOT Context Anchors and should not seed any direction inheritance.
A Data Provider is a factual assertion about queue length, wait time, or movement ("пусто", "30 машин", "стоїмо", "проїхав за годину").
Ignore as noise (no data, no anchor): messages that are punctuation-only, single reaction words ("майже", "ок"), meta-references to earlier messages ("читайте вище"), or sarcastic/joke replies containing no actual queue/time/movement information.
If a later message in the same short exchange explicitly retracts or corrects an earlier reply ("перепрошую, не так прочитав", "помилився"), discard the retracted message's data entirely.
If an ambiguous data message (no direction token) is followed within a few messages by a clarifying question-and-answer that resolves its direction (e.g. someone asks "це куди?" and gets an answer), retroactively apply that resolved direction to the original message.
If a later message reveals that an entire preceding exchange was actually discussing a different checkpoint, retroactively discard any data already attributed from that exchange.

=== QUEUE LENGTH EXTRACTION ===
Extract `reported_queue_lengths` only for the PRE-BARRIER queue — cars still waiting to be processed. Location phrasing for this checkpoint is NOT limited to any fixed list; recognize any phrase describing cars waiting before processing, using unambiguous positional markers such as: "перед шлагбаумом", "до шлагбауму", "на заїзд", "перед нами", "в полі", "на блокпосту", "на посту", "поза територією", or a clear preposition relative to a landmark ("до кільця", "перед окко").
Do NOT extract counts describing cars already inside / past initial processing — "територія", "на території", "після шлагбауму", "в боксах", "всередині". These describe an area assumed to already be at capacity once any pre-barrier queue exists; they carry no separate extractable count.
If a count appears alongside "територія" (or similar) WITHOUT one of the unambiguous pre-barrier markers clarifying it's actually before the barrier, skip extraction rather than guess.

Convert vague approximate phrasing ("+~20", "коло 20", "плюс-мінус 20", "приблизно") into a best single integer and set `is_approximate = true`. If a message admits incomplete visibility ("не бачу точно", "далі не видно"), still extract the stated number with `is_approximate = true`.
Vague-magnitude words with NO number at all ("багато", "величезна", "аж від [landmark]", "кілька", "пару") do not populate `reported_queue_lengths` — treat these as qualitative signals for `movement_state` only.
Distance-based estimates (km via navigator, e.g. "1.3км показує навігатор") must be IGNORED entirely — never convert distance to an estimated car count.

Do not emit more than one entry for the same source message with identical or near-identical values. Only emit multiple entries from a single message when it
explicitly describes multiple genuinely distinct segments (see additive checkpoint behavior) — never duplicate a single reported figure into two list entries.

=== CROSSING TIME EXTRACTION ===
Extract `reported_crossing_minutes` only for a FULLY completed crossing, stated as a total duration (e.g. "проїхали за 2 год" -> 120, "перетнули за 10 хвилин" -> 10). 
Do NOT extract: partial segments (e.g. time from arrival to entering the territory), ongoing/not-yet-finished waits ("вже стоїмо 1.5 год"), or general/typical-duration questions unrelated to right-now conditions ("скільки зазвичай займає перетин").
Ignore time reported to cross only one side of the checkpoint: "До Польщі - приїхали о 11:30, український кордон пройшли за 2 години, черга на польський рухається дуже повільно" - this means person crossed only half of the checkpoint, ignore this
time report. Only when a person says that both sides have been crossed ("пройшли обидва кордони", "пройшли два кордони", "пройшли український і польський кордон"), treat this as complete crossing time and extract it.

=== MOVEMENT STATE SIGNALS ===
Throughput/rate descriptions — whether phrased as an explicit rate ("10 машин в годину", "запуск раз в годину по 10-15") or as an example ("5 хвилин тому впустили 6", "з тих пір заїхали 15-17") — inform `movement_state` only; never extract these as a queue count or crossing time.
Qualitative severity words with no number ("капець", "жах", "все стоїть", "караул") also inform `movement_state` only (pushing toward slowdown/standstill).
Classify "standstill" ONLY when at least two independent messages corroborate a complete dead stop / no movement for an extended period. A single unconfirmed report of no movement should be classified as "slowdown" at most.

Extract data with maximum precision. When genuinely uncertain about direction, location, vehicle type, or checkpoint identity, output null rather than guessing — false nulls are far cheaper than false data.

=== MOVEMENT STATE INDEPENDENCE ===
Each direction's movement_state must be judged using ONLY messages that belong to that specific direction. A chat dominated by from_ukraine slowdown reports must NOT
influence to_ukraine's movement_state, and vice versa. If a direction has few or no qualifying messages, default that direction's movement_state to "normal" rather than
inferring it from the other direction's activity or the general tone of the chat.
"""

#   (c) it shares a sender ID (SENDER_ID-XXXX) with a nearby message that already has a
#       resolved direction (from an explicit token, reply chain, or rule (b) above) — the
#       same person is likely continuing their own earlier answer. Sender-match is a
#       supporting signal, not a standalone trigger: use it to reinforce or tie-break
#       between (a)/(b) when adjacency alone is ambiguous, but do not let it override a
#       clear, unambiguous direction token present elsewhere in the message itself. Two
#       different senders discussing the same topic are independent reports, not a
#       continuation, even if adjacent.


CHECKPOINT_SPECIFIC = """Checkpoint names: (Ukrainian side: {checkpoint_name}; {foreign_country_name} side: {counterpart_name})

=== FOREIGN CHECKPOINT FIREWALL ===
This checkpoint is {checkpoint_name} ({counterpart_name} on the {foreign_country_name} side). If a message attributes queue/time/movement data to a DIFFERENT named checkpoint, ignore that data. A message naming {checkpoint_name} or {counterpart_name} itself is NOT foreign and should be processed normally.
Do not treat mere mention of another checkpoint's name as disqualifying if no data is attributed to it (e.g. travel directions passing through another checkpoint are fine to ignore-but-not-flag).
Ignore any external links or references to other chats/channels (e.g. bare URLs, "дивись в іншому чаті") — never treat these as data and never attempt to resolve what they point to.
Ignore promotional, recruitment, or advertisement content, even if it happens to contain relevant keywords.

=== SEGMENTED QUEUE BEHAVIOR ===
{segment_instruction}

{landmark_block}
"""

LANDMARK_RULES = """=== LANDMARK REFERENCE for this checkpoint (recognize these and close variants/misspellings) ===
{landmark_reference}

Before extracting a landmark or count, first identify which clause of the message (if there are multiple) refers to passenger vehicles (cars/vans) specifically. Only extract data — including landmarks — from that clause. A number or landmark appearing only in a clause about a different vehicle type (coach bus, truck, pedestrian) must NEVER be borrowed into the passenger-vehicle report, even if it is the only landmark mentioned anywhere in the message.

- If a message says the passenger-vehicle queue reaches one of these landmarks and gives NO explicit car count, populate `landmark_mentioned` with the normalized label shown above and leave `value` null.
- If a message gives an explicit count AND references a landmark from the list, extract BOTH: the explicit count into `value`, and the landmark into `landmark_mentioned`. Do not discard the landmark just because a count is present — downstream logic uses both together.- If a landmark is mentioned that is not in this list, leave `landmark_mentioned` null.
- If the ONLY landmark reference in the message is attached to an excluded vehicle type (coach bus, truck, pedestrian), leave BOTH `value` and `landmark_mentioned` null for the passenger-vehicle report — do not reuse it.

Worked example (landmark misattribution to avoid):
Message: "Черга до Польщі легкові автомобілі територія, один автобус перед кільцем"
This describes TWO separate things: (1) passenger cars — status "територія" (on territory, no pre-barrier marker → null), and (2) a SEPARATE large coach bus ("автобус") positioned "перед кільцем" (before the roundabout).
The roundabout landmark belongs to the excluded bus, NOT the passenger-car queue.
Correct extraction for the passenger-vehicle report: value=null, landmark_mentioned=null.
INCORRECT: landmark_mentioned="roundabout" — this wrongly borrows the bus's landmark for the car queue.
"""

LOCATION_SEGMENTS = """=== LOCATION SEGMENTS AND LANDMARK HANDLING (direction-specific for this checkpoint) ===
FROM_UKRAINE (outbound) is ADDITIVE — this direction stages cars across multiple distinct pre-barrier points to avoid clutter at the barrier itself. Recognize these
segments for FROM_UKRAINE only (recognize these and close variants/misspellings):
- "staging" — matches: блокпост, блок пост, блок-пост, на блокпосту, на посту, в полі, на полі, поле
- "barrier" — matches: шлагбаум, перед шлагбаумом, до шлагбауму, світлофор, перед світлофором, до світлофора

If a message specifies which segment a count describes, populate `location_segment` with the NORMALIZED label shown above (e.g. "staging", not the raw text from the
message). If the segment mentioned doesn't clearly match one of these, leave `location_segment` null rather than inventing a new label — an unmatched segment name
is safer treated as unknown than as a new, unrecognized bucket.

If ONE message reports counts at multiple distinct pre-barrier locations (e.g. 'в полі' and 'перед шлагбаумом'), extract EACH as a SEPARATE entry in the list
(same source_message_id for both) — these are components of one physical queue, not competing estimates.

TO_UKRAINE (inbound) is CONTINUOUS — this direction has a single, unstaged queue. Do NOT apply `location_segment` to any to_ukraine message; leave it null always.
For to_ukraine, a count anchored to ANY named reference point (a landmark, junction, store, or other physical marker not explicitly configured below) should be treated as
INCOMPLETE and MUST be ignored. Only a bare, unanchored count (no reference point mentioned) should be treated as a complete to_ukraine observation.

For to_ukraine, treat landmark-anchoring as a property of the INDIVIDUAL message only — do not inherit "anchored to a landmark" from an earlier message in the same reply
chain or thread. Only ignore a count if the SAME message stating that count also names a reference point. A reply giving a number in response to "roughly how many
cars?" is a fresh, standalone estimate and should be extracted normally, even if an earlier message in the same thread mentioned a landmark.
"""

def build_prompt(
    checkpoint_config: dict[str, Any],
) -> tuple[str, str]:
    checkpoint_id = checkpoint_config["checkpoint_id"]
    checkpoint_name = checkpoint_config["display_name"]
    counterpart_name = checkpoint_config["foreign_name"]
    config_matrix = checkpoint_config.get("config_matrix", {})
    ai_heuristics = config_matrix.get("ai_heuristics", {})
    landmarks = ai_heuristics.get("landmark_mapping", {})
    segment_mode = ai_heuristics.get("segment_mode", "additive")

    prefix = checkpoint_id.split('_')[0] if '_' in checkpoint_id else ""
    foreign_country_name = {
        "PL": "Polish",
        "MD": "Moldovan",
        "SK": "Slovakian",
        "RO": "Romanian",
        "HU": "Hungarian"
    }.get(prefix, "neighboring country")

    if segment_mode == "additive":
        segment_instruction = LOCATION_SEGMENTS
    else:
        segment_instruction = (
            "This checkpoint has one continuous queue; landmark/location references just mark rough distance "
            "points along the same line, they do not describe separate segments. If a message mentions multiple "
            "location references, extract the single most specific/complete figure given, not a sum."
        )

    if landmarks:
        landmark_block = LANDMARK_RULES.format(
            landmark_reference="\n".join(
                f'- "{label}" — matches: {", ".join(variants)}'
                for label, variants in landmarks.items()
            )
        )
    else:
        landmark_block = ""
    
    return SYSTEM_PROMPT_TEMPLATE, CHECKPOINT_SPECIFIC.format(
        checkpoint_name=checkpoint_name,
        counterpart_name=counterpart_name,
        foreign_country_name=foreign_country_name,
        segment_instruction=segment_instruction,
        landmark_block=landmark_block,
    ).strip()
