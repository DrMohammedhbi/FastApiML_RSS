from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import feedparser
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime
import os

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

# New route to serve index.html
@app.get("/app", response_class=HTMLResponse)
async def read_index():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse("index.html")

@app.get("/")
async def root():
    return {
        "message": "RSS News Reader API",
        "endpoints": {
            "/app": "Web UI Interface",
            "/rss": "Fetch RSS feed by URL",
            "/rss/multiple": "Fetch multiple RSS feeds"
        }
    }

@app.get("/rss", response_model=RSSFeed)
async def get_rss_feed(url: HttpUrl, limit: Optional[int] = 10):
    """
    Fetch and parse an RSS feed from the provided URL
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
    """
    results = []
    
    for url in urls:
        try:
            feed = feedparser.parse(str(url))
            
            if feed.entries or not feed.bozo:
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
                    ).dict()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rss:app", host="127.0.0.1", port=8000, reload=True)
