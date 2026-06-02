from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class RouteRequest(BaseModel):
    origin: str
    destination: str

class RouteResponse(BaseModel):
    recommended_route: List[str]
    estimated_time: float

@app.post("/optimize", response_model=RouteResponse)
async def optimize_route(req: RouteRequest):
    # TODO: nanti diisi dengan logika optimasi
    return RouteResponse(recommended_route=[req.origin, "中转", req.destination], estimated_time=120.0)

@app.get("/health")
def health():
    return {"status": "ok"}