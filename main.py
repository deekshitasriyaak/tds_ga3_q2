from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import urllib.request

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")

class CommentRequest(BaseModel):
    comment: str

def get_sentiment(comment: str):
    url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
    data = json.dumps({"inputs": comment}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())

    # Result is a list of [{label, score}], pick highest score
    scores = result[0]
    best = max(scores, key=lambda x: x["score"])
    label = best["label"].lower()

    # Map model labels to your format
    label_map = {
        "positive": ("positive", 5),
        "negative": ("negative", 1),
        "neutral":  ("neutral",  3),
    }

    sentiment, rating = label_map.get(label, ("neutral", 3))
    return sentiment, rating

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")
    try:
        sentiment, rating = get_sentiment(request.comment)
        return JSONResponse(
            content={"sentiment": sentiment, "rating": rating},
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")