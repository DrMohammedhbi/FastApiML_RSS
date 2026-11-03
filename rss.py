from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import feedparser
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime

app = FastAPI(title="RSS News Reader API")

class RSSItem(BaseModel):
    title: str
    link: str
    published: Optional[str] = None
    summary: Optional[str] = None
    author: Optional[str] = None

class RSSFeed(BaseModel):
    feed_title: str
    feed_link: str
    items: List[RSSItem]

@app.get("/")
async def root():
    return {
        "message": "RSS News Reader API",
        "endpoints": {
            "/rss": "Fetch RSS feed by URL",
            "/rss/multiple": "Fetch multiple RSS feeds"
        }
    }

@app.get("/rss", response_model=RSSFeed)
async def get_rss_feed(url: HttpUrl, limit: Optional[int] = 10):
    """
    Fetch and parse an RSS feed from the provided URL
    
    - **url**: RSS feed URL
    - **limit**: Maximum number of items to return (default: 10)
    """
    try:
        feed = feedparser.parse(str(url))
        
        if feed.bozo:  # bozo bit indicates feed parsing issues
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse RSS feed: {feed.bozo_exception}"
            )
        
        items = []
        for entry in feed.entries[:limit]:
            items.append(RSSItem(
                title=entry.get('title', 'No title'),
                link=entry.get('link', ''),
                published=entry.get('published', None),
                summary=entry.get('summary', ''),
                author=entry.get('author', None)
            ))
        
        return RSSFeed(
            feed_title=feed.feed.get('title', 'Unknown Feed'),
            feed_link=feed.feed.get('link', ''),
            items=items
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching RSS feed: {str(e)}")

@app.post("/rss/multiple")
async def get_multiple_feeds(urls: List[HttpUrl], limit: Optional[int] = 5):
    """
    Fetch multiple RSS feeds at once
    
    - **urls**: List of RSS feed URLs
    - **limit**: Maximum number of items per feed (default: 5)
    """
    results = []
    
    for url in urls:
        try:
            feed = feedparser.parse(str(url))
            
            if not feed.bozo:
                items = []
                for entry in feed.entries[:limit]:
                    items.append(RSSItem(
                        title=entry.get('title', 'No title'),
                        link=entry.get('link', ''),
                        published=entry.get('published', None),
                        summary=entry.get('summary', ''),
                        author=entry.get('author', None)
                    ))
                
                results.append({
                    "url": str(url),
                    "status": "success",
                    "feed": RSSFeed(
                        feed_title=feed.feed.get('title', 'Unknown Feed'),
                        feed_link=feed.feed.get('link', ''),
                        items=items
                    )
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

# Example usage:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)