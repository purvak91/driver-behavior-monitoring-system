"""
FastAPI Backend for the Driver Behavior Monitoring System.
Provides REST endpoints for the vision pipeline to push telemetry,
and for the frontend dashboard to fetch leaderboard/live data.
"""
import json
import asyncio
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import engine, get_db, Base
from models import Driver, TrackingEvent, Violation

# Create all tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Driver Behavior Monitoring API")

# Allow frontend dev server to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────── WebSocket Manager ────────────────────────────
class ConnectionManager:
    """Manages active WebSocket connections for live feed."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()


# ──────────────────────────── Pydantic Schemas ─────────────────────────────
class TelemetryPayload(BaseModel):
    plate_number: str
    track_id: int = 0
    confidence: float = 0.0

class ViolationPayload(BaseModel):
    plate_number: str
    violation_type: str
    points_deducted: float = 5.0
    description: str = ""

class DriverResponse(BaseModel):
    id: int
    plate_number: str
    safety_score: float
    total_sightings: int
    last_seen_at: str
    violation_count: int

    class Config:
        from_attributes = True

class StatsResponse(BaseModel):
    total_drivers: int
    total_events: int
    total_violations: int
    avg_safety_score: float
    top_plate: str


# ──────────────────────────── REST Endpoints ───────────────────────────────

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Driver Behavior Monitoring System API — Phase 4",
    }


@app.post("/api/telemetry")
async def receive_telemetry(payload: TelemetryPayload, db: Session = Depends(get_db)):
    """
    Called by vision_pipeline.py every time a new plate is discovered.
    Creates or updates the Driver record and logs a TrackingEvent.
    """
    driver = db.query(Driver).filter(Driver.plate_number == payload.plate_number).first()

    if not driver:
        driver = Driver(
            plate_number=payload.plate_number,
            safety_score=100.0,
            total_sightings=1,
        )
        db.add(driver)
        db.commit()
        db.refresh(driver)
    else:
        driver.total_sightings += 1
        driver.last_seen_at = datetime.now(timezone.utc)
        db.commit()

    event = TrackingEvent(
        driver_id=driver.id,
        confidence=payload.confidence,
    )
    db.add(event)
    db.commit()

    # Broadcast to all connected WebSocket clients
    await manager.broadcast({
        "type": "new_detection",
        "plate_number": driver.plate_number,
        "safety_score": driver.safety_score,
        "total_sightings": driver.total_sightings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "status": "ok",
        "plate_number": driver.plate_number,
        "total_sightings": driver.total_sightings,
        "safety_score": driver.safety_score,
    }


@app.post("/api/violation")
async def report_violation(payload: ViolationPayload, db: Session = Depends(get_db)):
    """Report a traffic violation for a specific plate."""
    driver = db.query(Driver).filter(Driver.plate_number == payload.plate_number).first()
    if not driver:
        return {"status": "error", "message": "Driver not found in registry."}

    violation = Violation(
        driver_id=driver.id,
        violation_type=payload.violation_type,
        points_deducted=payload.points_deducted,
        description=payload.description,
    )
    db.add(violation)

    driver.safety_score = max(0, driver.safety_score - payload.points_deducted)
    db.commit()

    await manager.broadcast({
        "type": "violation",
        "plate_number": driver.plate_number,
        "violation_type": payload.violation_type,
        "points_deducted": payload.points_deducted,
        "new_score": driver.safety_score,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "status": "ok",
        "plate_number": driver.plate_number,
        "new_safety_score": driver.safety_score,
    }


@app.get("/api/leaderboard", response_model=List[DriverResponse])
def get_leaderboard(limit: int = 20, db: Session = Depends(get_db)):
    """Returns drivers sorted by safety_score descending (highest = safest)."""
    drivers = db.query(Driver).order_by(Driver.safety_score.desc()).limit(limit).all()
    results = []
    for d in drivers:
        v_count = db.query(Violation).filter(Violation.driver_id == d.id).count()
        results.append(DriverResponse(
            id=d.id,
            plate_number=d.plate_number,
            safety_score=d.safety_score,
            total_sightings=d.total_sightings,
            last_seen_at=d.last_seen_at.isoformat() if d.last_seen_at else "",
            violation_count=v_count,
        ))
    return results


@app.get("/api/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Aggregated statistics for the dashboard header."""
    total_drivers = db.query(Driver).count()
    total_events = db.query(TrackingEvent).count()
    total_violations = db.query(Violation).count()

    from sqlalchemy import func
    avg_score = db.query(func.avg(Driver.safety_score)).scalar() or 0.0
    top_driver = db.query(Driver).order_by(Driver.safety_score.desc()).first()

    return StatsResponse(
        total_drivers=total_drivers,
        total_events=total_events,
        total_violations=total_violations,
        avg_safety_score=round(avg_score, 1),
        top_plate=top_driver.plate_number if top_driver else "N/A",
    )


@app.get("/api/recent-activity")
def get_recent_activity(limit: int = 15, db: Session = Depends(get_db)):
    """Returns the most recent tracking events for the live feed panel."""
    events = (
        db.query(TrackingEvent)
        .order_by(TrackingEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    result = []
    for e in events:
        driver = db.query(Driver).filter(Driver.id == e.driver_id).first()
        result.append({
            "id": e.id,
            "plate_number": driver.plate_number if driver else "UNKNOWN",
            "timestamp": e.timestamp.isoformat(),
            "confidence": e.confidence,
            "safety_score": driver.safety_score if driver else 0,
        })
    return result

@app.delete("/api/driver/{driver_id}")
def delete_driver(driver_id: int, db: Session = Depends(get_db)):
    """Admin endpoint to entirely remove a driver and history."""
    driver = db.query(Driver).filter(Driver.id == driver_id).first()
    if not driver:
        return {"status": "error", "message": "Driver not found."}
    
    # Delete relationships manually to avoid constraint errors
    db.query(TrackingEvent).filter(TrackingEvent.driver_id == driver_id).delete()
    db.query(Violation).filter(Violation.driver_id == driver_id).delete()
    db.delete(driver)
    db.commit()

    return {"status": "ok", "message": f"Driver {driver_id} records purged."}


# ──────────────────────────── WebSocket Endpoint ───────────────────────────
@app.websocket("/api/live-feed")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
