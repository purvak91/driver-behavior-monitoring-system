"""
SQLAlchemy ORM models for the Driver Behavior Monitoring System.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from database import Base


class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, unique=True, index=True, nullable=False)
    safety_score = Column(Float, default=100.0)
    total_sightings = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    events = relationship("TrackingEvent", back_populates="driver")
    violations = relationship("Violation", back_populates="driver")


class TrackingEvent(Base):
    __tablename__ = "tracking_events"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    confidence = Column(Float, default=0.0)

    driver = relationship("Driver", back_populates="events")


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=False)
    violation_type = Column(String, nullable=False)  # e.g. "speeding", "signal_jump", "wrong_lane"
    points_deducted = Column(Float, default=5.0)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    description = Column(String, default="")
    frame_path = Column(String, default="")

    driver = relationship("Driver", back_populates="violations")
