This project is aimed at scaping Telegram groups about border crossings and extracting the information about the queue length, as well as the estimated time to cross the border.
scaper.py extracts messages from chats and saves them into local sqlite, llm_extractor.py sends messages to Gemini API for extraction of queue and time info and saves the results into Supabase.

Configuration files:

- scraper_config.toml - configuration for scraper.py
- llm_extractor_config.toml - configuration for llm_extractor.py

Config matrix file:

- config_matrix.py - configuration matrix for both scraper.py and llm_extractor.py, contains checkpoint information (handles, etc.)

llm_extractor.py also uses Nakordoni API for comparison of queue and time info with official data.
