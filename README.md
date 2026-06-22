## Border Bot Scraper: Border Bot Telegram Scraper and LLM Predictor

Data preparation pipeline for [BorderBot](https://t.me/ua_border_traffic_bot) built to track and crowdsource border crossing wait times.

## High-Level Architecture & Tech Stack

**Infrastructure:** Telegram Client.

**Compute:** Python Scraper and LLM pipeline.

**Database:** Supabase (PostgreSQL) DB shared with the bot, SQLite DB to keep Telegram messages between the runs.

## Key Features

1. Scans Telegram chats for key signals about queue length.

2. Calls nakordoni.eu API to have an independent baseline data.

3. Uses LLM to parse Telegram chats and predict how long it will take to cross a border.

4. Uses Supabase to store the data and predictions.

## Quick Start / Deployment Guide

1. Rename .env.example to .env and fill in the values
2. Create required folders:
    1. mkdir logs
    1. mkdir db
3. Run docker compose up -d
