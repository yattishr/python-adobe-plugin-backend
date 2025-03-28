import io
import os
import json
import asyncio
import random
import time
from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import instructor
import google.generativeai as genai
import google.ai.generativelanguage as glm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configure CORS middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Setting up the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Define Pydantic models
class Section(BaseModel):
    heading: str
    content: str

class AnalysisResponse(BaseModel):
    title: str
    summary: str
    sections: List[Section]
    conclusion: str

# Create instructor client
# client = instructor.from_gemini(
#     client=genai.GenerativeModel(model_name="gemini-1.5-pro"),
#     mode=instructor.Mode.GEMINI_JSON,
# )

client = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    tools=[translateText, transformText]
)

# Semaphore to limit the number of concurrent requests
MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Returns the appropriate tone based on the creativity level.
def get_tone(creativity_level: int) -> str:
    """
    Returns the appropriate tone based on the creativity level.
    """
    creativity_mapping = [
        (30, "formal"),
        (60, "conversational and friendly"),
        (90, "Shakespearean"),
        (100, "Pirate-like")
    ]
    
    for max_level, tone in creativity_mapping:
        if creativity_level <= max_level:
            return tone

    raise ValueError("Creativity level must be between 0 and 100")


def read_csv_data(file: UploadFile, sample_size=1000):
    try:
        contents = file.file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(buffer)
        return df.sample(min(sample_size, len(df)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")

def dataframe_to_text(df: pd.DataFrame):
    return df.to_csv(index=False)

async def send_data_to_gemini(data: str, description: str, creativity_level: int, max_retries=3):
    # print the incoming Creativity level
    logger.info(f"Incoming Creativity level: {creativity_level}")

    tone = get_tone(creativity_level)
    logger.info(f"Logging tone: {tone}")

    logger.info(f"Using tone: {tone}")

    system_prompt = """You are an expert data analyst who fully understands how to present data in story form
    that is easier for users to understand. Your role is to create a compelling story based on
    the data that is provided to highlight key aspects and components from the data.
    In addition, you will highlight any identifiable trends you recognize from the data and present it
    in a relatable story that the user can relate to.

    Your task is to analyze the data and provide an aggregated view of the data in a compelling story 
    based on the data that is provided, using a {} tone.    

    IMPORTANT: Ensure that your response follows proper JSON formatting. Do NOT include extra words,
    pirate slang, emojis, or anything that breaks the structure. Keep it fun but structured.

    Structure your response in the following JSON format:
    {{
        "title": "A catchy title for your analysis",
        "summary": "A brief summary of the key findings",
        "sections": [
            {{
                "heading": "Section heading",
                "content": "Section content"
            }},
            ...
        ],
        "conclusion": "A concluding paragraph"
    }}""".format(tone)  # Ensure dynamic tone is correctly inserted

    prompt = f"Analyze the following CSV data:\n```\nDescription of the data: {description}\n```\n{data}\n```\n{system_prompt}"

    initial_delay = 2
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending the prompt to the model (attempt {attempt + 1})...")
            response = client.messages.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_model=AnalysisResponse
            )
            logger.info("Response completed.")
            return response
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise HTTPException(status_code=500, detail="Failed to parse Gemini response as JSON")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            if attempt < max_retries - 1:
                wait_time = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Exiting.")
                raise HTTPException(status_code=500, detail=f"Error in Gemini API: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_csv(file: UploadFile = File(...), description: str = Form(...), creativity_level: int = Form(...)):    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    async with semaphore:
        try:
            df = read_csv_data(file, sample_size=1000)
            data = dataframe_to_text(df)
            print("Logging incoming creativity_level: ", creativity_level)
            response = await send_data_to_gemini(data, description, creativity_level)
            return response
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise HTTPException(status_code=500, detail="Failed to parse Gemini response as JSON")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if "429" in str(e):
                raise HTTPException(status_code=429, detail="API rate limit exceeded. Please try again later.")
            raise HTTPException(status_code=500, detail=f"Error in Gemini API: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)