# Crypto Twitter Bot Documentation

## Overview
This documentation covers a Twitter bot system designed to automatically generate and manage crypto-related content, handle user interactions, and engage with relevant conversations on Twitter. The bot uses AI-powered content generation and maintains scheduled posting capabilities.

## Core Components

### 1. TwitterAgent (main.py)
The central orchestrator that initializes and manages all bot components.

Key Features:
- Initializes the bot with personality configuration
- Manages scheduled operations
- Coordinates between different handlers
- Handles crypto content updates
- Runs periodic operational cycles

Configuration:
```dotenv
TWITTER_BEARER_TOKEN
TWITTER_API_KEY
TWITTER_API_SECRET
TWITTER_ACCESS_TOKEN
TWITTER_ACCESS_TOKEN_SECRET
OPENROUTER_API_KEY
COINGECKO_DEMO_API_KEY
EXA_API_KEY
NEWSDATA_API_KEY
```

### 2. ContentGenerator (content_generator.py)
Handles the generation of various types of content using AI models.

Supported Models for post_model_name:
- microsoft/wizardlm-2-8x22b
- nousresearch/hermes-3-llama-3.1-405b
- mistralai/mistral-nemo
- meta-llama/llama-3.1-70b-instruct

Methods:
- `generate_post()`: Creates new posts based on current events and context
- `generate_comment()`: Generates comments for existing tweets
- `generate_response()`: Creates responses to mentions
- `_format_text()`: Ensures content meets Twitter's character limits

### 3. CryptoNewsWorkflow (crypto_scraper.py)
Manages the collection and processing of crypto-related news and information.

Workflow Steps:
1. Retrieves trending crypto information
2. Fetches detailed coin information
3. Obtains positive company information
4. Schedules posts based on aggregated data

### 4. ScheduleManager (scheduler.py)
Manages the scheduling and timing of posts.

Features:
- Maintains pending and completed schedules
- Sorts schedules by time
- Handles schedule additions and removals
- Tracks overdue and future events

### 5. MentionResponder (mention_handler.py)
Handles responses to mentions and interactions.

Key Features:
- Processes mentions within a specified lookback period
- Generates contextual responses
- Maintains response history
- Filters mentions based on relevance

### 6. PostHandler (post_handler.py)
Manages the posting of scheduled content.

Features:
- Checks for due posts
- Generates and posts content
- Updates schedule status
- Handles posting errors

### 7. TweetEngager (tweet_engager.py)
Proactively engages with relevant tweets based on configured buzzwords.

Features:
- Searches for relevant tweets
- Generates contextual responses
- Maintains engagement history
- Handles rate limits and errors


### 8. All the Other Tools and Agents

#### Reply Context Agent
Purpose: Analyzes the context of replies and mentions
- Created via `create_reply_context_agent()`
- Uses model: "gpt-4o-mini"
- Provides contextual understanding for generating appropriate responses

#### Reply Composer Agent
Purpose: Composes reply content
- Created via `create_reply_composer_agent()`
- Uses configurable post model
- Generates contextually appropriate replies to mentions and interactions

#### Comment Context Agent
Purpose: Analyzes the context of comments
- Created via `create_comment_context_agent()`
- Uses model: "gpt-4o-mini"
- Provides understanding for generating relevant comments

#### Comment Composer Agent
Purpose: Composes comment content
- Created via `create_comment_composer_agent()`
- Uses configurable post model
- Generates engaging comments on relevant posts

#### Post Generator Agent
Purpose: Generates original posts
- Created via `create_post_generator_agent()`
- Uses configurable post model
- Creates original content based on current events and context

#### Trending Crypto Agent
Purpose: Retrieves trending cryptocurrency information
- Created via `create_trending_crypto_agent()`
- Requires COINGECKO_DEMO_API_KEY
- Fetches current crypto market trends and data

#### Deep Coin Info Agent
Purpose: Provides detailed cryptocurrency information
- Created via `create_deep_coin_info_agent()`
- Requires EXA_API_KEY
- Retrieves in-depth analysis and information about specific cryptocurrencies

#### Company Info Agent
Purpose: Gathers company-related information
- Created via `create_company_info_agent()`
- Requires EXA_API_KEY and NEWSDATA_API_KEY
- Collects relevant company news and information

#### Schedule Agent
Purpose: Manages post scheduling
- Created via `create_schedule_agent()`
- Uses ScheduleTool for scheduling operations
- Handles timing and organization of scheduled posts

#### Schedule Tool
Purpose: Provides scheduling functionality
- Used by Schedule Agent
- Manages the actual scheduling operations
- Interfaces with ScheduleManager for schedule tracking


## Setup and Configuration

1. Environment Setup:
   ```bash
    pip install -r requirements.txt
   ```

2. Configure Environment Variables:
   Create a `.env` file with the following:
   ```
   TWITTER_BEARER_TOKEN=your_bearer_token
   TWITTER_API_KEY=your_api_key
   TWITTER_API_SECRET=your_api_secret
   TWITTER_ACCESS_TOKEN=your_access_token
   TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
   OPENROUTER_API_KEY=your_openrouter_key
   COINGECKO_DEMO_API_KEY=your_coingecko_key
   EXA_API_KEY=your_exa_key
   NEWSDATA_API_KEY=your_newsdata_key
   ```



## Error Handling

The system implements comprehensive error handling:
- API rate limit management
- Connection error recovery
- Content generation fallbacks
- Schedule management recovery

## Monitoring and Maintenance

1. Logging:
   - System uses Python's logging module
   - Logs are formatted with timestamps and log levels
   - Critical operations are logged for debugging

2. Performance Monitoring:
   - Track API usage
   - Monitor response times
   - Track engagement metrics

3. Regular Maintenance:
   - Update API keys as needed
   - Monitor and update buzzwords

## Future Improvements

1. Planned Enhancements:
   - Firebase integration for persistent storage
   - Enhanced tweet relevance filtering
   - Improved content generation algorithms
   - Advanced scheduling capabilities

2. Potential Additions:
   - Analytics dashboard
   - Multi-account support
   - Advanced sentiment analysis
   - Enhanced error recovery mechanisms
Here's the command to install your dependencies from your requirements.txt file:

```sh
pip install -r requirements.txt
```

## Caution
Don't remove the test.json file from your project directory—make sure it remains in place, as it’s needed for testing and configuration purposes.Also make sure that it's path is set properly in the .env file
