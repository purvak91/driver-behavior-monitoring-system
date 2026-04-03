from fastapi import FastAPI

app = FastAPI(title="Driver Behavior Monitoring API")

@app.get("/")
def read_root():
    return {
        "status": "Application is running", 
        "message": "Welcome to the Reward-Based Driver Behavior Monitoring System API"
    }
