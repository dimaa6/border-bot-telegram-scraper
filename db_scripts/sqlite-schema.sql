-- Create the local cache table with a composite primary key
CREATE TABLE IF NOT EXISTS message_log (
    checkpoint_id TEXT,
    message_id INTEGER,
    reply_to_msg_id INTEGER,
    sender_id INTEGER,
    message_text TEXT,
    recorded_at TIMESTAMP, -- Stored as ISO8601 string text (YYYY-MM-DD HH:MM:SS)
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (checkpoint_id, message_id)
);

-- Indexing for rapid chronological lookups when assembling the AI prompt
CREATE INDEX IF NOT EXISTS idx_checkpoint_time 
ON message_log (checkpoint_id, recorded_at);
