#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

app = FastAPI(
    title="Safe Work Permit API",
    description="Simple backend for Aramco-style Safe Work Permit system",
    version="0.1.0",
)

# ===============================
# Pydantic models
# ===============================

class HazardControlSuggestion(BaseModel):
    hazards: List[str]
    controls: List[str]


class PermitBase(BaseModel):
    job_description: str = Field(..., example="Hot work on pump P-101 in gas plant area")
    location: str = Field(..., example="GOSP-4, Area B")
    permit_type: List[str] = Field(
        default_factory=list,
        example=["Hot Work", "Confined Space Entry"]
    )
    contractor: Optional[str] = Field(None, example="XYZ Contracting Co.")
    area_owner: Optional[str] = Field(None, example="Operations Dept.")
    equipment_code: Optional[str] = Field(None, example="P-101")
    hazards: List[str] = Field(default_factory=list)
    controls: List[str] = Field(default_factory=list)
    gas_test_required: bool = False
    status: str = Field("draft", example="draft")  # draft / issued / closed


class PermitCreate(PermitBase):
    pass


class PermitUpdate(BaseModel):
    job_description: Optional[str] = None
    location: Optional[str] = None
    permit_type: Optional[List[str]] = None
    contractor: Optional[str] = None
    area_owner: Optional[str] = None
    equipment_code: Optional[str] = None
    hazards: Optional[List[str]] = None
    controls: Optional[List[str]] = None
    gas_test_required: Optional[bool] = None
    status: Optional[str] = None


class Permit(PermitBase):
    id: int
    created_at: datetime
    updated_at: datetime


class GasTest(BaseModel):
    permit_id: int
    o2: float
    lel: float
    h2s: float
    co: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ===============================
# Fake in-memory storage
# (replace with DB in production)
# ===============================

permits_db: Dict[int, Permit] = {}
gas_tests_db: List[GasTest] = []
_next_id = 1


def _get_next_id() -> int:
    global _next_id
    val = _next_id
    _next_id += 1
    return val


# ===============================
# Simple "AI" hazard generator
# (Rule-based stub you can later
# replace with real ML)
# ===============================

def generate_hazards_and_controls(description: str) -> HazardControlSuggestion:
    desc = description.lower()
    hazards = set()
    controls = set()

    if any(k in desc for k in ["hot work", "welding", "cutting", "grinding"]):
        hazards.update(["Fire/explosion", "Burns from sparks"])
        controls.update([
            "Fire watch assigned",
            "Fire extinguisher in place",
            "Hot work permit displayed",
            "Remove flammables within 15m"
        ])

    if any(k in desc for k in ["confined space", "vessel entry", "tank entry"]):
        hazards.update(["Oxygen deficiency", "Toxic gas exposure", "Engulfment"])
        controls.update([
            "Confined space entry permit",
            "Continuous gas monitoring",
            "Standby man assigned",
            "Rescue plan prepared"
        ])

    if any(k in desc for k in ["electrical", "panel", "cable", "switchgear"]):
        hazards.update(["Electrical shock", "Arc flash"])
        controls.update([
            "LOTO applied and verified",
            "Insulated tools used",
            "Arc-rated PPE",
        ])

    if any(k in desc for k in ["height", "scaffold", "roof", "elevation"]):
        hazards.update(["Fall from height", "Falling objects"])
        controls.update([
            "Full body harness with double lanyard",
            "Scaffold tagged and inspected",
            "Barricade area below"
        ])

    # Default generic hazards
    if not hazards:
        hazards.update([
            "Slips, trips and falls",
            "Manual handling injuries"
        ])
        controls.update([
            "Keep area clear and tidy",
            "Use proper lifting technique"
        ])

    return HazardControlSuggestion(
        hazards=sorted(list(hazards)),
        controls=sorted(list(controls)),
    )


# ===============================
# Routes
# ===============================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ai/hazards", response_model=HazardControlSuggestion)
def ai_hazards(payload: Dict[str, Any]):
    description = payload.get("job_description", "")
    return generate_hazards_and_controls(description)


@app.post("/permits", response_model=Permit)
def create_permit(permit: PermitCreate):
    pid = _get_next_id()
    now = datetime.utcnow()

    # If hazards/controls empty, auto-generate
    if not permit.hazards and not permit.controls:
        ai = generate_hazards_and_controls(permit.job_description)
        hazards = ai.hazards
        controls = ai.controls
    else:
        hazards = permit.hazards
        controls = permit.controls

    full = Permit(
        id=pid,
        created_at=now,
        updated_at=now,
        hazards=hazards,
        controls=controls,
        **permit.dict(exclude={"hazards", "controls"})
    )
    permits_db[pid] = full
    return full


@app.get("/permits", response_model=List[Permit])
def list_permits(status: Optional[str] = None):
    vals = list(permits_db.values())
    if status:
        vals = [p for p in vals if p.status == status]
    return sorted(vals, key=lambda p: p.created_at, reverse=True)


@app.get("/permits/{permit_id}", response_model=Permit)
def get_permit(permit_id: int):
    if permit_id not in permits_db:
        raise HTTPException(status_code=404, detail="Permit not found")
    return permits_db[permit_id]


@app.put("/permits/{permit_id}", response_model=Permit)
def update_permit(permit_id: int, update: PermitUpdate):
    if permit_id not in permits_db:
        raise HTTPException(status_code=404, detail="Permit not found")
    stored = permits_db[permit_id]
    data = update.dict(exclude_unset=True)

    updated = stored.copy(update=data)
    updated.updated_at = datetime.utcnow()
    permits_db[permit_id] = updated
    return updated


@app.post("/gas-tests", response_model=GasTest)
def add_gas_test(gas: GasTest):
    if gas.permit_id not in permits_db:
        raise HTTPException(status_code=404, detail="Permit not found for gas test")
    gas_tests_db.append(gas)
    return gas


@app.get("/gas-tests/{permit_id}", response_model=List[GasTest])
def get_gas_tests(permit_id: int):
    return [g for g in gas_tests_db if g.permit_id == permit_id]
