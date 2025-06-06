from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from summarizers import extractive_summarizer, abstractive_summarizer

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummaryRequest(BaseModel):
    text: str
    summary_type: str = "quick"  # "quick" or "smart"
    length: str = "medium"       # "short", "medium", or "long"

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    try:
        if request.summary_type == "quick":
            summary = extractive_summarizer(request.text, length=request.length)
        else:
            summary = abstractive_summarizer(request.text, length=request.length)
        
        return {"summary": summary}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)