<div align="center">
  <h1>👁️ Operations Overwatch (CCTV Video Analytics)</h1>
  <p><em>Autonomous VLM-Powered Surveillance & Spatial Tracking Engine</em></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?style=for-the-badge&logo=huggingface)](https://huggingface.co/)
  [![NVIDIA LocateAnything](https://img.shields.io/badge/NVIDIA-LocateAnything--3B-76B900.svg?style=for-the-badge&logo=nvidia)](https://huggingface.co/nvidia/LocateAnything-3B)
  [![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)
  [![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-000000.svg?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)

  **[🔴 LIVE DEMO: Try Operations Overwatch on Hugging Face Spaces](https://huggingface.co/spaces/rudyxx07/CCTV-Operations-Overwatch)**
</div>

<br/>

> **The Problem:** Traditional CCTV monitoring relies on rigid object detection models (YOLO, Faster-RCNN) that are strictly limited to the specific classes they were trained on (e.g., "person", "car"). If you want to detect "a person slouching on an office chair" or "a partially occluded chair behind a desk," you have to manually label thousands of images and retrain the entire network.
>
> **The Solution (Operations Overwatch):** A 100% dynamic, prompt-driven **Vision-Language Surveillance Pipeline**. Instead of rigid classes, this pipeline deploys a 3-Billion parameter Multimodal VLM (`LocateAnything-3B`) that understands complex natural language predicates. Paired with a zero-parameter mathematical tracking algorithm (`OC-SORT`), the system flawlessly maintains spatial continuity across frames, tracking anything you can describe in English.

---

## ⚡ The "Secret Sauce" (Why This is Elite AI Engineering)

This is not a generic YOLO tutorial script. This is a highly optimized, enterprise-grade **Spatial Intelligence Workflow** built to run state-of-the-art Multimodal AI on consumer-grade hardware (sub-8GB VRAM).

### 🧠 The Brains: Parallel-Decoded Vision Language Models (`LocateAnything-3B`)
This system completely abandons legacy bounding-box regression. Powered by NVIDIA's `LocateAnything-3B`, it uses a Qwen-VL architecture augmented with **Parallel Box Decoding**. It understands intricate queries (like *"person sitting at a desk partially hidden behind a monitor"*) and emits normalized spatial coordinate tokens in parallel.

### 🗜️ The Optimization: 8-Bit Native Quantization (`bitsandbytes`)
Running a 3-Billion parameter multimodal foundation model usually requires expensive A100 GPUs. We engineered the pipeline using HuggingFace `transformers` and `bitsandbytes` to load the model in **8-bit precision**. This violently compresses the 6GB model footprint down to ~3GB, allowing it to run flawlessly on an entry-level RTX 5050 Laptop GPU without sacrificing sub-pixel coordinate accuracy.

### 🎯 The Memory: Observation-Centric Tracking (`OC-SORT`)
VLMs are incredibly smart but computationally expensive, making 60 FPS real-time inference impossible on edge devices. Instead of querying the VLM every frame, we query it periodically and pass the coordinates to an **OC-SORT (Observation-Centric SORT)** algorithm. This zero-parameter, purely mathematical Kalman Filter patches the fatal flaws of traditional tracking by relying on "virtual trajectories" during occlusions, running entirely on the CPU in fractions of a millisecond.

### 🛡️ The Guardrails: PAS Hallucination Filtering
Generative AI can hallucinate objects that aren't there. To make this production-ready for security environments, the pipeline implements a **PAS (Prelim Attention Score) Filter**. If the VLM generates text that lacks strict structural coordinate formatting (`<box><x><y>`), the output is mathematically penalized and gated, ensuring zero false-positive bounding boxes are rendered.

### 👁️‍🗨️ The Interface: Cloud-Native Batch Analytics (`Gradio` + `Hugging Face ZeroGPU`)
A command-line output isn't enough for surveillance. The system spins up a dynamic **Gradio** web interface for video analysis. 

*Technical Note: To effectively showcase this pipeline practically on enterprise cloud infrastructure (like Hugging Face Spaces' shared A100 GPUs), the architecture utilizes a `Video Upload -> Process -> Download` batch system. This intentional design choice circumvents the strict 120-second timeout limits of serverless GPU functions that would otherwise abruptly terminate continuous RTSP streams.*

---

## 🚀 Getting Started (Dual Deployment)

You can run this pipeline either locally on your own GPU, or deploy it directly to a free Hugging Face Space!

### Option 1: Local Execution (RTX Laptops / Desktop GPUs)
Ensure you have Python 3.10+ installed, then install the heavily optimized requirement stack (including `bitsandbytes` to load the model in 8-bit mode on consumer GPUs):

```bash
pip install -r requirements.txt
```

Run the Gradio application locally:
```bash
python app.py
```
Open the provided `http://127.0.0.1:7860` link in your browser.

### Option 2: Hugging Face Spaces Deployment (ZeroGPU A100s)
You can deploy this repository exactly as-is to a Hugging Face Space.
1. Create a new Space on Hugging Face (SDK: `Gradio`).
2. Push this repository's code to the Space.
3. Hugging Face's **ZeroGPU** will automatically intercept the `@spaces.GPU` decorators in `app.py`, routing your video processing to a massive A100 GPU cluster completely for free!