from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the given comment and respond ONLY with valid JSON in this exact format: "
                        '{"sentiment": "positive", "rating": 5} '
                        "Rules:\n"
                        "- sentiment must be exactly one of: positive, negative, neutral\n"
                        "- rating must be an integer 1-5 where 5=highly positive, 4=positive, 3=neutral, 2=negative, 1=highly negative\n"
                        "- positive sentiment -> rating 4 or 5\n"
                        "- neutral sentiment -> rating 3\n"
                        "- negative sentiment -> rating 1 or 2\n"
                        "No explanation, no extra text, just the JSON object."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        result = json.loads(response.choices[0].message.content)

        # Validate the fields
        if result.get("sentiment") not in ["positive", "negative", "neutral"]:
            raise ValueError("Invalid sentiment value")
        if not isinstance(result.get("rating"), int) or not (1 <= result["rating"] <= 5):
            raise ValueError("Invalid rating value")

        return JSONResponse(
            content={"sentiment": result["sentiment"], "rating": result["rating"]},
            media_type="application/json"
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")