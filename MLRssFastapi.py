from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import feedparser
from typing import List, Optional, Dict
from pydantic import BaseModel, HttpUrl
from datetime import datetime

app = FastAPI(title="RSS News Reader API with ML")

# Optional: Try importing transformers and torch, handle gracefully if missing
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class MLModels:
    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers/Torch not found. ML features will be disabled.")
            return
        
        try:
            print("Loading Sentiment Analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("Sentiment model loaded successfully!")
        except Exception as e:
            print(f"Failed to load sentiment model: {e}")
            self.sentiment_analyzer = None

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if not self.sentiment_analyzer or not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5}
        
        try:
            text = text[:512]
            result = self.sentiment_analyzer(text)[0]
            return result
        except Exception:
            return {"label": "NEUTRAL", "score": 0.5}

# Initialize models at startup
ml_models = None

@app.on_event("startup")
async def load_models():
    """Load ML models when the app starts"""
    global ml_models
    print("Initializing ML helper...")
    ml_models = MLModels()

class RSSItem(BaseModel):
    title: str
    link: str
    published: Optional[str] = None
    summary: Optional[str] = None
    author: Optional[str] = None
    sentiment: Optional[Dict] = None

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
            "/rss/simple": "Fetch RSS feed without ML analysis"
        }
    }

@app.get("/rss", response_model=RSSFeed)
async def get_rss_feed(
    url: HttpUrl,
    limit: Optional[int] = 10,
    analyze: bool = True
):
    """
    Fetch and parse an RSS feed with ML analysis
    """
    try:
        feed = feedparser.parse(str(url))
        
        if feed.bozo and not feed.entries:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse RSS feed: {feed.bozo_exception}"
            )
        
        items = []
        sentiments = []
        
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
                sentiment = ml_models.analyze_sentiment(text_content)
                item_data["sentiment"] = sentiment
                sentiments.append(sentiment['label'])
            
            items.append(RSSItem(**item_data))
        
        analysis_summary = None
        if analyze and sentiments:
            analysis_summary = {
                "total_articles": len(items),
                "sentiment_distribution": {
                    "positive": sentiments.count("POSITIVE"),
                    "negative": sentiments.count("NEGATIVE")
                },
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

from fastapi.responses import HTMLResponse, FileResponse
import os

@app.get("/app", response_class=HTMLResponse)
async def serve_ml_app():
    if not os.path.exists("indexML.html"):
        raise HTTPException(status_code=404, detail="index2.html file not found")
    return FileResponse("indexML.html")

@app.get("/rss/simple")
async def get_rss_feed_simple(url: HttpUrl, limit: Optional[int] = 10):
    """
    Fetch RSS feed without ML analysis (faster)
    """
    try:
        feed = feedparser.parse(str(url))
        
        if feed.bozo and not feed.entries:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
