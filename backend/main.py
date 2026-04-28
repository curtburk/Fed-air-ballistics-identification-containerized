"""
NAWSCL Range Surveillance - Aircraft Classification System
Powered by HP ZGX Nano AI Station

A Vision Language Model demo for aerial test range imagery analysis.
Classifies aircraft as manned or unmanned, identifies platform type,
and generates structured analysis reports.

Uses Qwen3-VL-8B-Instruct-FP8 served via vLLM for image understanding
and report generation in a single model.
"""

import os
import io
import base64
import random
import string
from datetime import datetime, timezone
from pathlib import Path
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8090/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "/models/Qwen3-VL-8B-Instruct-FP8")

# Known aircraft platforms at NAWSCL
KNOWN_PLATFORMS = {
    "manned": {
        "F/A-18E/F Super Hornet": {
            "category": "MANNED - STRIKE FIGHTER",
            "role": "Multi-role strike fighter, weapons integration and software testing",
            "unit": "VX-9 Vampires / VX-31 Dust Devils"
        },
        "F-35C Lightning II": {
            "category": "MANNED - 5TH GEN STRIKE FIGHTER",
            "role": "Stealth carrier-based strike fighter, operational test and evaluation",
            "unit": "VX-9 Vampires"
        },
        "F-35B Lightning II": {
            "category": "MANNED - 5TH GEN STOVL",
            "role": "Short takeoff / vertical landing strike fighter",
            "unit": "Various test squadrons"
        },
        "EA-18G Growler": {
            "category": "MANNED - ELECTRONIC WARFARE",
            "role": "Electronic attack aircraft, EW systems testing",
            "unit": "VX-9 Vampires"
        },
        "AV-8B Harrier": {
            "category": "MANNED - V/STOL ATTACK",
            "role": "Vertical/short takeoff attack aircraft",
            "unit": "Various test detachments"
        }
    },
    "unmanned": {
        "MQ-1 Predator": {
            "category": "UNMANNED - ISR/STRIKE UAS",
            "role": "Medium-altitude, long-endurance ISR and precision strike",
            "unit": "Test and evaluation squadrons"
        },
        "MQ-8 Fire Scout": {
            "category": "UNMANNED - ROTARY-WING UAS",
            "role": "Unmanned helicopter for ISR and precision targeting",
            "unit": "Navy UAS test detachments"
        },
        "MQ-9 Reaper": {
            "category": "UNMANNED - HUNTER-KILLER UAS",
            "role": "Long-endurance armed ISR platform",
            "unit": "Test and evaluation squadrons"
        },
        "MQ-25 Stingray": {
            "category": "UNMANNED - CARRIER-BASED TANKER",
            "role": "Unmanned carrier-based aerial refueling",
            "unit": "Navy UAS programs"
        },
        "BQM-74 Chukar": {
            "category": "UNMANNED - TARGET DRONE",
            "role": "Subsonic aerial target for weapons testing",
            "unit": "NAWSCL Range Operations"
        }
    }
}

# Test range areas
RANGE_AREAS = {
    "baker_range": {
        "name": "Baker Range (R-2508N)",
        "lat_range": (35.7, 36.1),
        "lon_range": (-117.9, -117.4),
        "landmarks": ["Echo Range", "George Range", "Superior Valley"]
    },
    "charlie_range": {
        "name": "Charlie Range (R-2508S)",
        "lat_range": (35.3, 35.7),
        "lon_range": (-117.8, -117.3),
        "landmarks": ["Mojave B Range", "Randsburg Wash", "Searles Valley"]
    },
    "coso_range": {
        "name": "Coso Range Complex",
        "lat_range": (36.0, 36.3),
        "lon_range": (-117.9, -117.5),
        "landmarks": ["Coso Hot Springs", "Darwin Plateau", "Haiwee Reservoir"]
    },
    "armitage_field": {
        "name": "Armitage Airfield",
        "lat_range": (35.685, 35.695),
        "lon_range": (-117.695, -117.680),
        "landmarks": ["Main Runway 21/03", "NAF China Lake", "Ridgecrest"]
    },
    "superior_valley": {
        "name": "Superior Valley Range",
        "lat_range": (35.8, 36.0),
        "lon_range": (-117.5, -117.2),
        "landmarks": ["Panamint Valley", "Trona Pinnacles", "Searles Lake"]
    },
    "north_range": {
        "name": "North Range (R-2502)",
        "lat_range": (36.3, 36.8),
        "lon_range": (-117.8, -117.2),
        "landmarks": ["Owens Valley", "Darwin Hills", "Centennial Flat"]
    }
}

app = FastAPI(
    title="NAWSCL Range Surveillance - Aircraft Classification",
    description="VLM-powered aerial platform identification for test range operations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files from frontend directory
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "/app/frontend")
frontend_path = Path(FRONTEND_DIR)
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Async HTTP client for calling vLLM
http_client = httpx.AsyncClient(timeout=120.0)


# -- vLLM interaction --------------------------------------------------------

async def query_vlm(image_b64: str, prompt: str, max_tokens: int = 512) -> tuple:
    """Send an image + prompt to Qwen3-VL via vLLM's OpenAI-compatible API.
    Returns (content_text, usage_dict)."""
    try:
        response = await http_client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json={
                "model": VLLM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        return content, usage
    except Exception as e:
        logger.error(f"vLLM query failed: {e}")
        raise


async def analyze_image_with_vlm(image_b64: str) -> tuple:
    """
    Analyze image using Qwen3-VL with a single comprehensive prompt.
    Returns (structured_analysis_dict, usage_dict).
    """
    logger.info("Analyzing image with Qwen3-VL...")

    prompt = """Analyze this aerial or ground-level image of an aircraft. Answer each question on its own line in exactly this format:

AIRCRAFT_TYPE: [What specific aircraft is this? e.g., F/A-18E/F Super Hornet, F-35C Lightning II, MQ-1 Predator, MQ-8 Fire Scout, MQ-9 Reaper, or describe if unknown]
MANNED_UNMANNED: [Is this a MANNED or UNMANNED aircraft? Look for a cockpit/canopy for manned, or lack thereof for unmanned]
DESCRIPTION: [Brief description of what you see - airframe shape, wing configuration, engines, landing gear state]
WEAPONS_PAYLOAD: [What is visible under the wings or fuselage? Describe any weapons, sensors, pods, fuel tanks, or pylons]
FLIGHT_STATUS: [Is the aircraft in flight, on the ground, taxiing, or on a runway? Describe its current state]
MARKINGS: [Describe any visible markings, numbers, insignia, or paint scheme]

Be specific and concise. Base your answers only on what you can see in the image."""

    raw_response, usage = await query_vlm(image_b64, prompt, max_tokens=400)

    # Parse the structured response
    results = {
        "aircraft_type": "UNIDENTIFIED AIRCRAFT",
        "manned_unmanned": "UNDETERMINED",
        "description": "No description available",
        "weapons_payload": "Unknown",
        "flight_status": "Unknown",
        "markings": "None visible"
    }

    for line in raw_response.strip().split("\n"):
        line = line.strip()
        for key in results.keys():
            tag = key.upper() + ":"
            if line.upper().startswith(tag):
                value = line[len(tag):].strip()
                if value:
                    results[key] = value
                break

    logger.info(f"VLM analysis results: {results}")
    return results, usage


# -- Geolocation generation --------------------------------------------------

def generate_synthetic_coordinates(range_area: str) -> dict:
    """Generate realistic synthetic coordinates within the specified range area."""
    if range_area not in RANGE_AREAS:
        range_area = "baker_range"

    area_data = RANGE_AREAS[range_area]

    lat = random.uniform(*area_data["lat_range"])
    lon = random.uniform(*area_data["lon_range"])

    # Convert to degrees, minutes, seconds
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"

    lat_abs = abs(lat)
    lon_abs = abs(lon)

    lat_deg = int(lat_abs)
    lat_min = int((lat_abs - lat_deg) * 60)
    lat_sec = int(((lat_abs - lat_deg) * 60 - lat_min) * 60)

    lon_deg = int(lon_abs)
    lon_min = int((lon_abs - lon_deg) * 60)
    lon_sec = int(((lon_abs - lon_deg) * 60 - lon_min) * 60)

    # Generate MGRS-style grid reference (simplified)
    grid_zone = f"{int((lon + 180) / 6) + 1:02d}"
    grid_band = chr(ord('C') + int((lat + 80) / 8))
    grid_square = ''.join(random.choices(string.ascii_uppercase[:8], k=2))
    grid_easting = f"{random.randint(1000, 9999)}"
    grid_northing = f"{random.randint(1000, 9999)}"

    # Select nearby landmark
    landmark = random.choice(area_data["landmarks"])
    distance_nm = random.randint(2, 30)
    bearing = random.randint(0, 359)

    # Altitude (synthetic)
    altitude_ft = random.choice([
        random.randint(500, 2000),      # Low altitude
        random.randint(10000, 25000),   # Medium altitude
        random.randint(25000, 45000),   # High altitude
    ])

    return {
        "decimal": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "dms": f"{lat_deg}\u00b0{lat_min:02d}'{lat_sec:02d}\"{lat_dir}, {lon_deg}\u00b0{lon_min:02d}'{lon_sec:02d}\"{lon_dir}",
        "mgrs": f"{grid_zone}{grid_band} {grid_square} {grid_easting} {grid_northing}",
        "relative": f"{distance_nm}nm {bearing}\u00b0 from {landmark}",
        "region_name": area_data["name"],
        "altitude_ft": altitude_ft
    }


def generate_report_id() -> str:
    """Generate a realistic report identifier."""
    prefix = random.choice(["RANGE", "ISR", "TEVAL", "SURV"])
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = f"{random.randint(1, 9999):04d}"
    return f"{prefix}-{date_str}-{seq}"


# -- Aircraft classification -------------------------------------------------

def classify_aircraft(image_analysis: dict) -> tuple:
    """Determine manned/unmanned status and platform category from analysis.
    Returns (manned_unmanned, category, platform_match, confidence_note)."""
    all_text = " ".join(image_analysis.values()).lower()
    manned_unmanned_raw = image_analysis.get("manned_unmanned", "").upper()
    aircraft_type_raw = image_analysis.get("aircraft_type", "").lower()

    # Try to match a known platform
    platform_match = None
    for manned_status, platforms in KNOWN_PLATFORMS.items():
        for name, info in platforms.items():
            if name.lower() in aircraft_type_raw or name.lower().replace("/", "") in all_text:
                platform_match = {"name": name, **info, "manned_status": manned_status.upper()}
                break
        if platform_match:
            break

    # Determine manned/unmanned from VLM output or keyword matching
    if platform_match:
        manned_unmanned = platform_match["manned_status"]
        category = platform_match["category"]
    elif "unmanned" in manned_unmanned_raw or "unmanned" in all_text or any(
        kw in all_text for kw in ["uav", "drone", "uas", "mq-", "bqm-", "rq-"]
    ):
        manned_unmanned = "UNMANNED"
        category = "UNMANNED - UNCLASSIFIED UAS"
    elif "manned" in manned_unmanned_raw or any(
        kw in all_text for kw in ["cockpit", "canopy", "pilot", "crew", "f-18", "f/a-18", "f-35", "ea-18", "av-8"]
    ):
        manned_unmanned = "MANNED"
        category = "MANNED - UNCLASSIFIED AIRCRAFT"
    else:
        manned_unmanned = "UNDETERMINED"
        category = "UNCLASSIFIED AIRCRAFT"

    # Confidence note
    if platform_match:
        confidence_note = f"Positive identification: {platform_match['name']}"
    else:
        confidence_note = "Platform not positively identified from known NAWSCL inventory"

    return manned_unmanned, category, platform_match, confidence_note


async def generate_mission_assessment(image_b64: str, aircraft_type: str, manned_unmanned: str) -> tuple:
    """Use VLM to generate a contextual mission profile assessment. Returns (text, usage_dict)."""
    prompt = f"""Based on this image of a {manned_unmanned.lower()} aircraft identified as {aircraft_type}, provide a brief 2-sentence assessment of:
1. What mission profile this aircraft configuration suggests (weapons test, ISR sortie, training flight, etc.)
2. Whether any visible weapons, sensors, or payload configurations are noteworthy.
Be specific to what you observe in the image."""

    try:
        response, usage = await query_vlm(image_b64, prompt, max_tokens=150)
        return response, usage
    except Exception:
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def generate_recommendations(manned_unmanned: str, platform_match: dict, weapons_payload: str) -> str:
    """Generate contextual recommendations based on classification."""
    recommendations = []

    if manned_unmanned == "UNMANNED":
        recommendations.append("Verify UAS operating area clearance and airspace deconfliction")
        recommendations.append("Confirm ground control station link status and lost-link procedures")
        recommendations.append("Log flight track for range safety corridor compliance")
    elif manned_unmanned == "MANNED":
        recommendations.append("Verify flight plan against published range schedule")
        recommendations.append("Confirm IFF squawk and range frequency assignment")
    else:
        recommendations.append("PRIORITY: Determine manned/unmanned status for airspace management")
        recommendations.append("Request visual confirmation from range safety officer")

    # Payload-specific
    payload_lower = weapons_payload.lower()
    if any(kw in payload_lower for kw in ["missile", "bomb", "weapon", "ordnance", "munition"]):
        recommendations.append("Confirm weapons safety status and range-clear authorization")
        recommendations.append("Verify hot/cold range status before weapons release authorization")
    elif any(kw in payload_lower for kw in ["sensor", "pod", "camera", "eo", "ir"]):
        recommendations.append("Log sensor payload configuration for test data correlation")

    if platform_match:
        recommendations.append(f"Cross-reference with {platform_match.get('unit', 'assigned unit')} flight schedule")

    recommendations.append("Update range surveillance log and forward to Range Control")

    return "\n   - ".join(recommendations)


async def build_analysis_report(image_analysis: dict, image_b64: str, custom_instructions: str = "") -> tuple:
    """Build structured analysis report from VLM analysis. Returns (report_text, usage_dict)."""
    aircraft_type = image_analysis.get("aircraft_type", "UNIDENTIFIED AIRCRAFT")
    description = image_analysis.get("description", "No description available")
    weapons_payload = image_analysis.get("weapons_payload", "Unknown")
    flight_status = image_analysis.get("flight_status", "Unknown")
    markings = image_analysis.get("markings", "None visible")

    # Classify aircraft
    manned_unmanned, category, platform_match, confidence_note = classify_aircraft(image_analysis)

    # Get VLM mission assessment
    mission_assessment, assessment_usage = await generate_mission_assessment(image_b64, aircraft_type, manned_unmanned)
    if not mission_assessment:
        mission_assessment = "Unable to generate mission assessment from available imagery."

    # Generate recommendations
    recommendations_text = generate_recommendations(manned_unmanned, platform_match, weapons_payload)
    if custom_instructions:
        recommendations_text += f"\n   - {custom_instructions}"

    # Platform info if matched
    platform_info = ""
    if platform_match:
        platform_info = f"""
   Platform: {platform_match['name']}
   Role: {platform_match['role']}
   Unit: {platform_match['unit']}"""

    assessment = f"""1. PLATFORM IDENTIFICATION
   Classification: {manned_unmanned}
   Category: {category}{platform_info}

2. PHYSICAL CHARACTERISTICS
   Visual Description: {description.capitalize()}
   Markings: {markings.capitalize()}
   Weapons/Payload: {weapons_payload.capitalize()}

3. FLIGHT STATUS
   Current State: {flight_status.upper()}

4. MISSION PROFILE ASSESSMENT
   \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
   MANNED/UNMANNED: {manned_unmanned}
   \u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
   Assessment: {mission_assessment}

5. CONFIDENCE LEVEL: {"HIGH" if platform_match else "MODERATE"}
   {confidence_note}
   Assessment based on Qwen3-VL vision-language model analysis

6. RECOMMENDATIONS
   - {recommendations_text}"""

    return assessment, assessment_usage


# -- Image analysis pipeline -------------------------------------------------

async def analyze_image(image: Image.Image, range_area: str, custom_instructions: str = "") -> dict:
    """Full analysis pipeline: VLM analysis -> classification -> report generation."""

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if too large
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    # Convert to base64 for vLLM API
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Get structured analysis from VLM
    image_analysis, analysis_usage = await analyze_image_with_vlm(image_b64)

    # Build analysis report
    analysis, assessment_usage = await build_analysis_report(image_analysis, image_b64, custom_instructions)

    # Accumulate token usage across both VLM calls
    token_usage = {
        "prompt_tokens": analysis_usage.get("prompt_tokens", 0) + assessment_usage.get("prompt_tokens", 0),
        "completion_tokens": analysis_usage.get("completion_tokens", 0) + assessment_usage.get("completion_tokens", 0),
        "total_tokens": analysis_usage.get("total_tokens", 0) + assessment_usage.get("total_tokens", 0)
    }

    # Generate synthetic position data
    geo_data = generate_synthetic_coordinates(range_area)
    report_id = generate_report_id()
    capture_time = datetime.now(timezone.utc).strftime("%d %b %Y %H%MZ").upper()

    # Determine manned/unmanned for top-level field
    manned_unmanned, category, platform_match, _ = classify_aircraft(image_analysis)

    report = {
        "report_id": report_id,
        "classification": "UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY",
        "capture_time": capture_time,
        "manned_unmanned": manned_unmanned,
        "platform_category": category,
        "platform_match": platform_match["name"] if platform_match else None,
        "location": {
            "coordinates_dms": geo_data["dms"],
            "coordinates_decimal": geo_data["decimal"],
            "grid_reference": geo_data["mgrs"],
            "relative_position": geo_data["relative"],
            "range_area": geo_data["region_name"],
            "altitude_ft": geo_data["altitude_ft"]
        },
        "analysis": analysis,
        "raw_analysis": image_analysis,
        "token_usage": token_usage,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }

    return report


# -- Routes -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    index_path = Path(FRONTEND_DIR) / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>NAWSCL Range Surveillance</h1><p>index.html not found</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint - also verifies vLLM is responsive."""
    vllm_healthy = False
    try:
        resp = await http_client.get(f"{VLLM_BASE_URL.replace('/v1', '')}/health", timeout=5.0)
        vllm_healthy = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "healthy" if vllm_healthy else "degraded",
        "vllm_server": "ready" if vllm_healthy else "not ready",
        "model": "Qwen3-VL-8B-Instruct-FP8",
        "inference_engine": "vLLM",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/ranges")
async def get_ranges():
    """Get available test range areas."""
    return {
        "ranges": [
            {"id": k, "name": v["name"], "landmarks": v["landmarks"]}
            for k, v in RANGE_AREAS.items()
        ]
    }


@app.post("/api/analyze")
async def analyze_endpoint(
    image: UploadFile = File(...),
    range_area: str = Form("baker_range"),
    custom_instructions: str = Form("")
):
    """Analyze an uploaded image and generate an aircraft classification report."""

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await image.read()

    if len(image_data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 20MB)")

    try:
        img = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    try:
        report = await analyze_image(img, range_area, custom_instructions)
        return JSONResponse(content=report)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    host_ip = os.environ.get("HOST_IP", "")
    print("\n" + "=" * 60)
    print("  NAWSCL Range Surveillance - Aircraft Classification")
    print("  Qwen3-VL-8B-Instruct-FP8 | HP ZGX Nano | vLLM")
    print("=" * 60)
    if host_ip:
        print(f"\n  \u27a1  http://{host_ip}:{PORT}")
    else:
        print(f"\n  \u27a1  http://localhost:{PORT}")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up HTTP client."""
    await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
