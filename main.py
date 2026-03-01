from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load model once at startup
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")
    try:
        result = sentiment_pipeline(request.comment)[0]
        label = result["label"].lower()
        score = result["score"]

        if label == "positive":
            sentiment = "positive"
            rating = 5 if score > 0.9 else 4
        elif label == "negative":
            sentiment = "negative"
            rating = 1 if score > 0.9 else 2
        else:
            sentiment = "neutral"
            rating = 3

        return JSONResponse(
            content={"sentiment": sentiment, "rating": rating},
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")