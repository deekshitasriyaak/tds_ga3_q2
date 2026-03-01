from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")

    try:
        blob = TextBlob(request.comment)
        polarity = blob.sentiment.polarity
        # polarity is between -1 (negative) and +1 (positive)

        if polarity > 0.1:
            sentiment = "positive"
            rating = 5 if polarity > 0.5 else 4
        elif polarity < -0.1:
            sentiment = "negative"
            rating = 1 if polarity < -0.5 else 2
        else:
            sentiment = "neutral"
            rating = 3

        return JSONResponse(
            content={"sentiment": sentiment, "rating": rating},
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")