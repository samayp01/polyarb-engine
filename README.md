# Topic - Real-Time News Alerts

A lightweight system that turns user-defined topics into real-time push alerts.

- Ingests news from RSS, APIs, and social feeds
- Uses embeddings to match articles to user topics
- Scores articles by recency, source quality, uniqueness, and relevance
- Sends push notifications via FCM when important matches are found
- Runs every 5 minutes with spam protection and deduplication

Built as a minimal POC to validate the idea: users get timely, high-signal alerts.
