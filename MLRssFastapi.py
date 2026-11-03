from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import feedparser
from typing import List, Optional, Dict
from pydantic import BaseModel, HttpUrl
from datetime import datetime
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

app = FastAPI(title="RSS News Reader API with ML")

# Initialize ML models (loaded once at startup)
class MLModels:
    def __init__(self):
        # Sentiment Analysis Model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Text Summarization Model
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        
        # Zero-shot Classification for Topic Detection
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Topic categories for news classification
        self.topics = [
            "Technology", "Business", "Science", "Politics",
            "Entertainment", "Sports", "Health", "Finance"
        ]
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5}
        
        # Truncate text if too long
        text = text[:512]
        result = self.sentiment_analyzer(text)[0]
        return result
    
    def summarize_text(self, text: str, max_length: int = 130) -> str:
        """Generate summary of text"""
        if not text or len(text.split()) < 50:
            return text
        
        try:
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
            return summary
        except:
            return text[:200] + "..."
    
    def classify_topic(self, text: str) -> Dict:
        """Classify text into topic categories"""
        if not text or len(text.strip()) == 0:
            return {"topic": "Unknown", "confidence": 0.0}
        
        text = text[:512]
        result = self.classifier(text, self.topics)
        return {
            "topic": result['labels'][0],
            "confidence": result['scores'][0],
            "all_scores": dict(zip(result['labels'], result['scores']))
        }

# Initialize models at startup
ml_models = None

@app.on_event("startup")
async def load_models():
    """Load ML models when the app starts"""
    global ml_models
    print("Loading ML models...")
    ml_models = MLModels()
    print("ML models loaded successfully!")

class RSSItem(BaseModel):
    title: str
    link: str
    published: Optional[str] = None
    summary: Optional[str] = None
    author: Optional[str] = None
    # ML Analysis fields
    sentiment: Optional[Dict] = None
    topic: Optional[Dict] = None
    ai_summary: Optional[str] = None

class RSSFeed(BaseModel):
    feed_title: str
    feed_link: str
    items: List[RSSItem]
    analysis_summary: Optional[Dict] = None

@app.get("/")
async def root():
    return {
        "message": "RSS News Reader API with ML Analysis",
        "endpoints": {
            "/rss": "Fetch RSS feed by URL with ML analysis",
            "/rss/multiple": "Fetch multiple RSS feeds with ML analysis",
            "/rss/simple": "Fetch RSS feed without ML analysis"
        },
        "ml_features": {
            "sentiment_analysis": "Analyzes sentiment of each article",
            "topic_classification": "Classifies articles into topics",
            "ai_summarization": "Generates AI summaries of articles"
        }
    }

@app.get("/rss", response_model=RSSFeed)
async def get_rss_feed(
    url: HttpUrl,
    limit: Optional[int] = 10,
    analyze: bool = True,
    summarize: bool = False
):
    """
    Fetch and parse an RSS feed with ML analysis
    
    - **url**: RSS feed URL
    - **limit**: Maximum number of items to return (default: 10)
    - **analyze**: Perform sentiment and topic analysis (default: True)
    - **summarize**: Generate AI summaries (default: False, slower)
    """
    try:
        feed = feedparser.parse(str(url))
        
        if feed.bozo:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse RSS feed: {feed.bozo_exception}"
            )
        
        items = []
        sentiments = []
        topics = []
        
        for entry in feed.entries[:limit]:
            # Extract text for analysis
            text_content = entry.get('summary', '') or entry.get('title', '')
            
            item_data = {
                "title": entry.get('title', 'No title'),
                "link": entry.get('link', ''),
                "published": entry.get('published', None),
                "summary": entry.get('summary', ''),
                "author": entry.get('author', None)
            }
            
            # Perform ML analysis if requested
            if analyze and ml_models and text_content:
                # Sentiment Analysis
                sentiment = ml_models.analyze_sentiment(text_content)
                item_data["sentiment"] = sentiment
                sentiments.append(sentiment['label'])
                
                # Topic Classification
                topic = ml_models.classify_topic(text_content)
                item_data["topic"] = topic
                topics.append(topic['topic'])
                
                # AI Summarization (optional, slower)
                if summarize and len(text_content.split()) > 50:
                    item_data["ai_summary"] = ml_models.summarize_text(text_content)
            
            items.append(RSSItem(**item_data))
        
        # Generate overall analysis summary
        analysis_summary = None
        if analyze and sentiments:
            analysis_summary = {
                "total_articles": len(items),
                "sentiment_distribution": {
                    "positive": sentiments.count("POSITIVE"),
                    "negative": sentiments.count("NEGATIVE")
                },
                "top_topics": list(set(topics[:5])),
                "dominant_sentiment": max(set(sentiments), key=sentiments.count) if sentiments else "NEUTRAL"
            }
        
        return RSSFeed(
            feed_title=feed.feed.get('title', 'Unknown Feed'),
            feed_link=feed.feed.get('link', ''),
            items=items,
            analysis_summary=analysis_summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching RSS feed: {str(e)}")

@app.get("/rss/simple")
async def get_rss_feed_simple(url: HttpUrl, limit: Optional[int] = 10):
    """
    Fetch RSS feed without ML analysis (faster)
    """
    try:
        feed = feedparser.parse(str(url))
        
        if feed.bozo:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse RSS feed: {feed.bozo_exception}"
            )
        
        items = []
        for entry in feed.entries[:limit]:
            items.append({
                "title": entry.get('title', 'No title'),
                "link": entry.get('link', ''),
                "published": entry.get('published', None),
                "summary": entry.get('summary', ''),
                "author": entry.get('author', None)
            })
        
        return {
            "feed_title": feed.feed.get('title', 'Unknown Feed'),
            "feed_link": feed.feed.get('link', ''),
            "items": items
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching RSS feed: {str(e)}")

@app.post("/rss/multiple")
async def get_multiple_feeds(
    urls: List[HttpUrl],
    limit: Optional[int] = 5,
    analyze: bool = True
):
    """
    Fetch multiple RSS feeds with ML analysis
    """
    results = []
    
    for url in urls:
        try:
            feed = feedparser.parse(str(url))
            
            if not feed.bozo:
                items = []
                for entry in feed.entries[:limit]:
                    text_content = entry.get('summary', '') or entry.get('title', '')
                    
                    item_data = {
                        "title": entry.get('title', 'No title'),
                        "link": entry.get('link', ''),
                        "published": entry.get('published', None),
                        "summary": entry.get('summary', ''),
                        "author": entry.get('author', None)
                    }
                    
                    if analyze and ml_models and text_content:
                        item_data["sentiment"] = ml_models.analyze_sentiment(text_content)
                        item_data["topic"] = ml_models.classify_topic(text_content)
                    
                    items.append(item_data)
                
                results.append({
                    "url": str(url),
                    "status": "success",
                    "feed": {
                        "feed_title": feed.feed.get('title', 'Unknown Feed'),
                        "feed_link": feed.feed.get('link', ''),
                        "items": items
                    }
                })
            else:
                results.append({
                    "url": str(url),
                    "status": "error",
                    "error": str(feed.bozo_exception)
                })
        except Exception as e:
            results.append({
                "url": str(url),
                "status": "error",
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})

# Example usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)