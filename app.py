from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def greet_json():
    return {"Project": "DrugTargetAI deployed successfully"}




