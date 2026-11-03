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

