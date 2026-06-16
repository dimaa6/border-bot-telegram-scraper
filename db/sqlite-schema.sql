-- Create the local cache table with a composite primary key
CREATE TABLE IF NOT EXISTS message_log (
    channel_username TEXT,
    message_id INTEGER,
    sender_id INTEGER,
    message_text TEXT,
    recorded_at TIMESTAMP, -- Stored as ISO8601 string text (YYYY-MM-DD HH:MM:SS)
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (channel_username, message_id)
);

-- Indexing for rapid chronological lookups when assembling the AI prompt
CREATE INDEX IF NOT EXISTS idx_channel_time 
ON message_log (channel_username, recorded_at);