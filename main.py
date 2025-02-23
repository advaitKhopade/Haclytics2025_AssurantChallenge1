import os
import uuid
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from predict_disaster import load_model, predict_disaster

# 1. Load environment variables from .env (if present)
load_dotenv()

# # 2. Retrieve the LLaMA API URL and model name from environment (with fallbacks)
# LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:8001/v1/chat/completions")
# LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "llama-2-7b-chat")

# 3. Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Load your existing vision model once at startup
model = load_model(model_path="best_model.pth", num_disaster_types=8)

def get_llama_recommendation(disaster_info: dict) -> str:
    """
    Calls a LLaMA API to generate actionable insights/recommendations.
    disaster_info is the dictionary returned by predict_disaster().
    Adjust the prompt as needed.
    """
    predicted_type = disaster_info.get("predicted_disaster_type", "unknown type")
    predicted_severity = disaster_info.get("predicted_severity", "unknown severity")

    # Construct a user-friendly prompt for LLaMA
    prompt_content = (
        f"You are an expert assistant. A user has uploaded an image showing a disaster. "
        f"The system identified the disaster type as '{predicted_type}' "
        f"with severity '{predicted_severity}'. "
        "Please provide actionable insights and protection recommendations tailored to the user's needs. "
        "Keep it concise and helpful."
    )

    # # The payload may vary depending on your LLaMA server’s API spec
    # payload = {
    #     "model": LLAMA_MODEL_NAME,  
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": prompt_content
    #         }
    #     ],
    #     "max_tokens": 200,
    #     "temperature": 0.7
    # }

    # try:
    #     response = requests.post(LLAMA_API_URL, json=payload)
    #     response.raise_for_status()
    #     data = response.json()
    #     # Adjust extraction based on your server’s output format
    #     recommendation_text = data["choices"][0]["message"]["content"]
    #     return recommendation_text.strip()
    # except Exception as e:
    #     print(f"Error calling LLaMA API: {e}")
    #     return "Unable to generate recommendation at this time."

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns the predicted
    disaster type, severity, plus LLaMA-based recommendation.
    """
    try:
        # Create a unique temp filename
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        temp_filepath = os.path.join("images", temp_filename)

        # Save file to disk
        with open(temp_filepath, "wb") as f:
            f.write(await file.read())

        # Perform prediction
        result = predict_disaster(temp_filepath, model=model)

        # Optionally remove temp file
        os.remove(temp_filepath)

        # Now get a recommendation from LLaMA
    #  llama_text = get_llama_recommendation(result)

        # Add the recommendation and a follow-up prompt to the result
     #   result["recommendation"] = llama_text
    #    result["follow_up_prompt"] = "Do you have any additional questions or clarifications?"

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
