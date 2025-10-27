"""
DeepSeek-OCR RunPod Handler
Simple, clean implementation for serverless deployment
Author: Based on existing handler pattern
"""

import runpod
import os
import io
import json
import re
import base64
import torch
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import AutoModel, AutoTokenizer
import google.generativeai as genai

# ============================================================================
# GLOBAL MODEL INITIALIZATION (runs once when container starts)
# ============================================================================


print("Initializing DeepSeek-OCR Model...")

# Check GPU availability first
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[OK] GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
else:
    print("[WARNING] No GPU detected!")

# Load DeepSeek-OCR model
try:
    import time
    start_time = time.time()
    
    MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
    print(f"[INFO] Downloading/Loading tokenizer from {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    print(f"[OK] Tokenizer loaded ({time.time() - start_time:.1f}s)")
    
    print(f"[INFO] Loading pre-downloaded model...")
    model_start = time.time()
    
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True  
    )
    model = model.eval()
    
    print(f"[OK] Model loaded ({time.time() - model_start:.1f}s)")
    print(f"[OK] Total initialization time: {time.time() - start_time:.1f}s")
    
except Exception as e:
    print(f"[ERROR] Error initializing DeepSeek-OCR model: {str(e)}")
    import traceback
    print(traceback.format_exc())
    raise

# Initialize Gemini API
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY not found in environment variables!")
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Use gemini-1.5-flash without system instruction for direct JSON output
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print(f"[OK] Gemini API configured (Model: gemini-1.5-flash, Key: {GEMINI_API_KEY[:10]}...)")
    
except Exception as e:
    print(f"[ERROR] Error initializing Gemini: {str(e)}")
    import traceback
    print(traceback.format_exc())
    raise

print("All models loaded. Ready to process PDFs!")



# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def pdf_to_images(pdf_bytes, dpi=200):
    """Convert PDF to images"""
    try:
        images = convert_from_bytes(
            pdf_bytes,
            dpi=dpi,
            fmt='png'
        )
        # Ensure RGB format
        images = [img.convert("RGB") for img in images]
        return images
    except Exception as e:
        raise RuntimeError(f"PDF conversion failed: {str(e)}")


def extract_markdown_from_images(images, output_path="/tmp/"):

    temp_image_path = f"{output_path}temp_image.jpg"
    
    if len(images) == 1:
        images[0].save(temp_image_path)
    else:
        images[0].save(temp_image_path)
    
    print(f"Processing {len(images)} page(s)...")
    

    model.infer(
        tokenizer,
        prompt="<image>\nConvert to structuredmarkdown.",
        image_file=temp_image_path,
        output_path=output_path,
        base_size=640,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=False
    )
    
    # Read the generated markdown file
    result_file = f"{output_path}result.mmd"
    with open(result_file, 'r') as f:
        markdown_text = f.read()
    
    print(f"Extracted {len(markdown_text)} characters")
    
    return markdown_text


def extract_json_with_gemini(markdown_text):
    extraction_prompt = f"""You are an expert load confirmation analyst with over 10 years of experience as a dispatcher and broker in the trucking industry. You are thoroughly familiar with all possible terminologies used in load confirmation, dispatch, and rate agreement documents—including synonyms and alternate phrasing (for instance, "commodity" can also appear as "goods" or "product"; "carrier pay" as "freight rate" or "total payment"; etc.). Your role is to ensure that not a single piece of critical information is missed, regardless of wording, layout variation, or format.

Your task is:

Given the following Markdown-formatted text extracted from a load confirmation, rate confirmation, or dispatch document:

Identify and extract every key data point required for carrier load execution and payment, even if fields are phrased differently or appear in varied order.

Ensure you cross-reference synonymous terms and do not overlook information if it appears under a different label.

Omit no critical details that would impact payment, scheduling, claims, compliance, or operational clarity.

Strictly avoid inventing or hallucinating values. If a field is not present in the document, output an empty string, array, or object as appropriate.

tip: Have total grasp of the whole extracted data, so try to understand what is what and correctly fit in the JSON structure even the text is far away from.

Present the output exactly in the following JSON structure (and nothing else—no commentary, explanation, or extra text):

OCR EXTRACTED TEXT:
{markdown_text}

REQUIRED JSON SCHEMA:
{{
  "load_details": {{
    "broker_name": "",
    "broker_mc_number": "",
    "load_confirmation_number": "",
    "order_number": "",
    "bol_number": "",
    "reference_numbers": [],
    "commodity": "",
    "weight": "",
    "piece_count": "",
    "temperature_requirements": "",
    "equipment_type": "",
    "total_miles": ""
  }},
  "financial": {{
    "base_rate": "",
    "total_carrier_pay": "",
    "accessorial_charges": {{}},
    "detention_rate": "",
    "detention_terms": "",
    "payment_terms": ""
  }},
  "pickup": {{
    "facility_name": "",
    "address": "",
    "city": "",
    "state": "",
    "zip": "",
    "date": "",
    "time_window": "",
    "contact_information": "",
    "reference_numbers": [],
    "special_instructions": ""
  }},
  "delivery": {{
    "facility_name": "",
    "address": "",
    "city": "",
    "state": "",
    "zip": "",
    "date": "",
    "time_window": "",
    "contact_information": "",
    "reference_numbers": [],
    "special_instructions": ""
  }},
  "driver_equipment": {{
    "driver_name": "",
    "driver_phone": "",
    "tractor_number": "",
    "trailer_number": "",
    "tractor_vin": ""
  }},
  "operational_requirements": {{
    "tracking_requirements": "",
    "communication_protocols": "",
    "loading_responsibility": "",
    "unloading_responsibility": ""
  }},
  "penalties_restrictions": {{
    "cancellation_fee": "",
    "rescheduling_fee": "",
    "late_delivery_penalty": "",
    "weekend_holiday_restrictions": ""
  }}
}}

Extract the information and respond with ONLY valid JSON. No markdown formatting, no code blocks, no explanations.
"""
    
    try:
        # Generate content with NO safety filters - direct JSON output only
        response = gemini_model.generate_content(
            extraction_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=4096,
                response_mime_type="application/json",
            )
        )
        
        # Get the response text directly
        extracted_text = response.text.strip()
        
        # Clean JSON response
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text.replace("```json", "").replace("```", "").strip()
        elif extracted_text.startswith("```"):
            extracted_text = extracted_text.replace("```", "").strip()
        
        # Parse and return JSON only
        extracted_data = json.loads(extracted_text)
        return extracted_data
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        return {"error": f"JSON parsing error: {str(e)}"}
    except Exception as e:
        print(f"Gemini API error: {e}")
        raise


# ============================================================================
# RUNPOD HANDLER
# ============================================================================

def handler(event):

    try:
        job_input = event["input"]

        if "base64_pdf" not in job_input:
            return {
                "success": False,
                "data": None,
                "error": "No PDF data provided. Please provide 'base64_pdf' in input."
            }

        dpi = job_input.get("dpi", 200)
        return_markdown = job_input.get("return_markdown", False)

        try:
            pdf_bytes = base64.b64decode(job_input["base64_pdf"])
            print(f"Decoded PDF ({len(pdf_bytes)} bytes)")
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Invalid base64 PDF data: {str(e)}"
            }
        
        # Step 2: Convert PDF to images
        images = pdf_to_images(pdf_bytes, dpi=dpi)
        print(f"Converted to {len(images)} images (DPI: {dpi})")
        
        # Step 3: Extract markdown using DeepSeek-OCR
        # Use /tmp/ for temporary files on RunPod
        markdown_text = extract_markdown_from_images(images, output_path="/tmp/")
        print(f"OCR completed ({len(markdown_text)} characters)")
        
        # Step 4: Extract JSON using Gemini
        extracted_data = extract_json_with_gemini(markdown_text)
        print(f"JSON extraction completed")
        
        # Prepare response
        response = {
            "success": True,
            "data": extracted_data,
            "error": None
        }
        
        # Optionally include markdown
        if return_markdown:
            response["markdown"] = markdown_text
        
        print("="*70)
        print("Processing complete!")
        print("="*70 + "\n")
        
        return response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }


# ============================================================================
# START RUNPOD SERVERLESS
# ============================================================================

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

