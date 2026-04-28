# NAWSCL Range Surveillance - Aircraft Classification System

**Powered by HP ZGX Nano AI Station**

Vision Language Model (VLM) demonstration for aerial test range imagery analysis with manned/unmanned aircraft classification and structured analysis report generation. Runs entirely on-premises with no cloud dependency.

---

## What This Demo Is

A range surveillance tool that analyzes aerial imagery to classify aircraft as manned or unmanned, identify the specific platform, and generate a structured analysis report including mission profile assessment, payload identification, and range safety recommendations.

The application runs **Qwen3-VL-8B-Instruct-FP8**, a vision-language model with FP8 quantization served via **vLLM** for high-performance GPU inference. A single model handles both image understanding and natural language report generation.

Each analysis displays **token consumption metrics** (prompt, completion, and total tokens), demonstrating exactly what the inference cost would be on a cloud API, and why on-premises inference eliminates those costs entirely.

### Known Platforms

**Manned Aircraft:** F/A-18E/F Super Hornet, F-35C Lightning II, EA-18G Growler, AV-8B Harrier

**Unmanned Aircraft:** MQ-1 Predator, MQ-8 Fire Scout, MQ-9 Reaper, MQ-25 Stingray, BQM-74 Chukar

---

## What It Proves to Customers

1. **Classified range imagery never leaves the platform.** Test and evaluation data stays within the secure environment. No cloud APIs, no network round-trips.

2. **Real-time classification without connectivity.** Functions in disconnected, denied, or intermittent (DDIL) environments. Zero dependency on external networks.

3. **Vision AI runs locally on HP hardware.** An 8-billion parameter multimodal model classifying aircraft in seconds, on hardware the customer owns and controls.

4. **No per-query costs.** The token counter shows exactly how many tokens each analysis consumes. On a cloud API, that is a billable event. On HP hardware, it is unlimited.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/curtburk/Fed-air-ballistics-identification-containerized.git
cd Fed-air-ballistics-identification-containerized

# 2. Download the model (~9GB)
./download_models.sh

# 3. Start the demo
./start.sh
```

The terminal will print a clickable URL with the host IP (e.g., `http://192.168.x.x:8000`). First startup takes 2-3 minutes for model loading.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| GPU | HP ZGX Nano with NVIDIA GB10 Grace Blackwell, or any NVIDIA GPU with 12GB+ VRAM |
| System Memory | 32GB+ recommended |
| Storage | ~20GB free (9GB model + container image) |
| OS | Ubuntu 22.04 or 24.04 LTS |
| Docker | Docker Engine + Docker Compose |
| NVIDIA Container Toolkit | `nvidia-ctk` for GPU passthrough |

### Verify NVIDIA Container Toolkit

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

---

## Architecture

```
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
|   HTML Frontend   |---->|  FastAPI Backend   |---->|  vLLM Server      |
|   (index.html)    |     |  (main.py :8000)   |     |  (Qwen3-VL :8090) |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
                              All inside one Docker container
```

**Frontend** -- Test range-themed HTML/CSS/JavaScript interface with image upload, range area selection, manned/unmanned classification badge, structured report display, and per-analysis token consumption metrics.

**Backend** -- FastAPI server that receives image uploads, base64-encodes them, and sends multimodal prompts to vLLM's OpenAI-compatible API. Handles platform identification, manned/unmanned classification, mission profile assessment, and token usage tracking across VLM calls.

**Inference Engine** -- vLLM serving Qwen3-VL-8B-Instruct-FP8 on GPU. FP8 quantization reduces memory usage to ~9GB while maintaining near-identical quality. Exposes an OpenAI-compatible `/v1/chat/completions` endpoint on internal port 8090.

**Containerization** -- Based on `nvcr.io/nvidia/vllm:26.01-py3`. The entrypoint script starts vLLM in the background, waits for model loading, then starts the FastAPI application.

---

## Directory Structure

```
china-lake-aircraft-classification/
|-- backend/
|   |-- main.py                     # FastAPI application
|   |-- requirements-docker.txt     # Slim runtime dependencies
|   +-- entrypoint.sh              # Container startup script
|-- frontend/
|   |-- index.html                  # Test range-themed web interface
|   |-- hp_logo.png                 # HP branding
|   +-- Navy-Emblem.png            # US Navy emblem
|-- sample-images/
|   |-- image-1.png     # Manned aircraft sample
|   |-- image-2.png     # Unmanned aircraft sample
|   +-- image-3.png         # Unmanned aircraft sample
|-- models/
|   +-- Qwen3-VL-8B-Instruct-FP8/ # Downloaded model (~9GB)
|-- Dockerfile                      # Based on NVIDIA vLLM container
|-- docker-compose.yml              # One-command startup with GPU
|-- start.sh                        # Launch script with IP detection
|-- download_models.sh              # Model download from HuggingFace
|-- .dockerignore
|-- .gitignore
+-- README.md
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/analyze` | POST | Analyze image and classify aircraft |
| `/api/health` | GET | Health check (includes vLLM status) |
| `/api/ranges` | GET | List available test range areas |

### POST /api/analyze

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "image=@MQ-1_Predator.png" \
  -F "range_area=baker_range" \
  -F "custom_instructions=Focus on payload configuration"
```

The response includes manned/unmanned classification and token usage:

```json
{
  "report_id": "RANGE-20260425-1234",
  "manned_unmanned": "UNMANNED",
  "platform_category": "UNMANNED - ISR/STRIKE UAS",
  "platform_match": "MQ-1 Predator",
  "analysis": "...",
  "token_usage": {
    "prompt_tokens": 1842,
    "completion_tokens": 347,
    "total_tokens": 2189
  }
}
```

---

## Test Range Areas

| Range | Designation | Key Landmarks |
|-------|-------------|---------------|
| **Baker Range** | R-2508N | Echo Range, George Range, Superior Valley |
| **Charlie Range** | R-2508S | Mojave B Range, Randsburg Wash, Searles Valley |
| **Coso Range Complex** | -- | Coso Hot Springs, Darwin Plateau, Haiwee Reservoir |
| **Armitage Airfield** | NAF China Lake | Main Runway 21/03, Ridgecrest |
| **Superior Valley Range** | -- | Panamint Valley, Trona Pinnacles, Searles Lake |
| **North Range** | R-2502 | Owens Valley, Darwin Hills, Centennial Flat |

---

## The Customer Conversation

**Opening:** "Let me show you what happens when you put a vision-language AI model on hardware that never needs to phone home, right here on your range."

**During Demo:** Upload sample aircraft images. Select the relevant range area. Walk through the 6-section analysis report: platform identification (with MANNED/UNMANNED badge), physical characteristics, flight status, mission profile assessment, confidence level, and recommendations. Point out the token count at the bottom.

**Key Messages:**
- "This image was classified by an 8-billion parameter multimodal AI, running locally"
- "The imagery never left this machine. No cloud, no API calls, no network required"
- "Manned or unmanned, the model identifies the platform and generates a structured report in seconds"
- "See the token count? On a cloud API, that analysis would cost money every time. Here, it is unlimited"

**Closing:** "This is one example. The same hardware runs any AI workload where your data cannot leave your environment."

---

## Stopping

```bash
docker compose down
```

---

## Troubleshooting

**Docker daemon not running**
```bash
sudo systemctl start docker
```

**Permission denied on Docker commands**
```bash
sudo usermod -aG docker $USER && newgrp docker
```

**Model not found at startup** -- Ensure `Qwen3-VL-8B-Instruct-FP8/` directory exists in `./models/`. Run `./download_models.sh` if missing.

**vLLM fails to start** -- Check GPU memory with `nvidia-smi`. The FP8 model requires ~9GB VRAM. Ensure no other GPU processes are consuming memory.

**Slow first analysis** -- Normal. The first image analysis after startup may take longer as vLLM warms up. Subsequent analyses are faster.

**Cannot connect from another machine** -- Verify the firewall allows port 8000:
```bash
sudo ufw allow 8000
```

---

## Hardware

This demo is designed for the **HP ZGX Nano AI Station** but the FP8 quantized model runs on any NVIDIA GPU with 12GB+ VRAM, including:

- HP ZGX Nano (NVIDIA GB10 Grace Blackwell, 128GB unified memory)
- NVIDIA RTX 4090 Laptop (16GB)
- NVIDIA RTX 3090/4090 Desktop (24GB)
- NVIDIA RTX 5090 Laptop (24GB)

---

## Security Notice

This demonstration generates **synthetic position data** for illustration purposes. No actual operational coordinates are used or inferred from imagery.

The classification banner displays `UNCLASSIFIED // FOR DEMONSTRATION PURPOSES ONLY // NAWSCL CHINA LAKE` to clearly indicate the demo nature of the application.

---

## License

Internal HP demo. Contact the HP ZGX Nano product team for access and distribution questions.
