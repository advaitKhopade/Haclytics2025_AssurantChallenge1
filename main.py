# main.py

import os
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from predict_disaster import load_model, predict_disaster

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS if your frontend is served from a different origin
# (Adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup (instead of reloading for every request)
model = load_model(model_path="best_model.pth", num_disaster_types=8)

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns the predicted
    disaster type and severity as JSON.
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

        # Remove temp file if desired
        os.remove(temp_filepath)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
