"""
DeepSeek-OCR RunPod Handler - FIXED VERSION
Addresses Gemini API response errors and markdown capture issues
"""

import runpod
import os
import io
import json
import re
import base64
import torch
import glob
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import AutoModel, AutoTokenizer
import google.generativeai as genai

# ============================================================================
# GLOBAL MODEL INITIALIZATION
# ============================================================================

print("Initializing DeepSeek-OCR Model...")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[OK] GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
else:
    print("[WARNING] No GPU detected!")

try:
    import time
    start_time = time.time()
    
    MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
    print(f"[INFO] Loading tokenizer from {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    print(f"[OK] Tokenizer loaded ({time.time() - start_time:.1f}s)")
    
    print(f"[INFO] Loading model...")
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
    print(f"[OK] Total initialization: {time.time() - start_time:.1f}s")
    
except Exception as e:
    print(f"[ERROR] Model initialization failed: {str(e)}")
    import traceback
    print(traceback.format_exc())
    raise

# Initialize Gemini
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"[OK] Gemini configured (Model: {GEMINI_MODEL}, Key: {GEMINI_API_KEY[:10]}...)")
    
except Exception as e:
    print(f"[ERROR] Gemini initialization failed: {str(e)}")
    raise

print("="*70)
print("All models loaded. Ready to process!")
print("="*70 + "\n")


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def pdf_to_images(pdf_bytes, dpi=200):
    """Convert PDF to images"""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt='png')
        images = [img.convert("RGB") for img in images]
        return images
    except Exception as e:
        raise RuntimeError(f"PDF conversion failed: {str(e)}")


def extract_markdown_from_images(images, output_path="/tmp/"):
    """Combine all PDF pages into one tall image and run eval_mode=True."""
    import uuid
    from PIL import Image

    unique_id = str(uuid.uuid4())[:8]
    temp_dir = f"{output_path}ocr_{unique_id}/"
    os.makedirs(temp_dir, exist_ok=True)
    print(f"[DEBUG] temp_dir: {temp_dir}")

    # Normalize widths and stack vertically
    widths = [im.width for im in images]
    max_w = max(widths)
    resized = []
    total_h = 0
    for im in images:
        if im.width != max_w:
            new_h = int(im.height * (max_w / im.width))
            im = im.resize((max_w, new_h), Image.BILINEAR)
        resized.append(im)
        total_h += im.height

    combined = Image.new("RGB", (max_w, total_h), color=(255, 255, 255))
    y = 0
    for im in resized:
        combined.paste(im, (0, y))
        y += im.height

    combined_path = f"{temp_dir}combined_load_confirmation.jpg"
    combined.save(combined_path, quality=95)
    print(f"[DEBUG] combined image saved: {combined_path} ({max_w}x{total_h})")

    # Single inference using eval_mode=True to get string output
    res = model.infer(
        tokenizer,
        prompt="<image>\nConvert to markdown.",
        image_file=combined_path,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        test_compress=False,
        eval_mode=True
    )
    text = str(res).strip()

    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"[WARNING] cleanup failed: {e}")

    print(f"Extracted {len(text)} characters (combined)")
    return text


def extract_json_with_gemini(markdown_text):
    """
    Extract JSON using Gemini with ROBUST error handling
    FIXED: Properly handles response.text errors and safety blocks
    """
    

    MAX_CHARS = 100000
    if len(markdown_text) > MAX_CHARS:
        print(f"[WARNING] Markdown too long ({len(markdown_text)} chars), truncating to {MAX_CHARS}")
        markdown_text = markdown_text[:MAX_CHARS] + "\n\n[TRUNCATED]"
    
    extraction_prompt = f"""You are an expert load confirmation analyst. Extract data from this trucking document.

OCR MARKDOWN:
{markdown_text}

Extract information into this EXACT JSON structure (output ONLY valid JSON, no markdown, no code blocks):

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

Rules:
- Extract ALL information present in the document
- If a field is missing, use empty string/array/object
- Output ONLY valid JSON, nothing else"""
    
    try:
        # Configure safety settings using enums (SDK-compliant)
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        response = gemini_model.generate_content(
            extraction_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
            safety_settings=safety_settings
        )
        
        # FIX: Robust response handling
        # Check if response was blocked
        if not response.candidates:
            print("[ERROR] Gemini blocked the response (no candidates)")
            return {
                "error": "Gemini safety filter blocked response",
                "prompt_feedback": str(response.prompt_feedback) if hasattr(response, 'prompt_feedback') else "Unknown"
            }
        
        # Check if candidate was blocked
        candidate = response.candidates[0]
        if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name == 'SAFETY':
            print(f"[ERROR] Gemini blocked due to safety: {candidate.safety_ratings}")
            return {
                "error": "Content blocked by Gemini safety filters",
                "safety_ratings": str(candidate.safety_ratings)
            }
        
        # Try to get text from response
        try:
            extracted_text = response.text.strip()
        except ValueError as e:
            # response.text failed, try accessing parts directly
            print(f"[WARNING] response.text failed: {e}")
            if candidate.content and candidate.content.parts:
                extracted_text = candidate.content.parts[0].text.strip()
            else:
                print("[ERROR] No valid text in response")
                return {
                    "error": "No valid text returned by Gemini",
                    "finish_reason": candidate.finish_reason.name if hasattr(candidate, 'finish_reason') else "Unknown"
                }
        
        # Clean JSON response (remove markdown code blocks if present)
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text.replace("```json", "").replace("```", "").strip()
        elif extracted_text.startswith("```"):
            extracted_text = extracted_text.replace("```", "").strip()
        
        # Parse JSON
        extracted_data = json.loads(extracted_text)
        return extracted_data
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        print(f"Raw response: {extracted_text[:500]}...")
        return {
            "error": f"JSON parsing error: {str(e)}",
            "raw_response": extracted_text[:1000]
        }
    except Exception as e:
        print(f"[ERROR] Gemini API error: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "error": f"Gemini error: {str(e)}"
        }


# ============================================================================
# RUNPOD HANDLER
# ============================================================================

def handler(event):
    """Main RunPod handler"""
    
    print("\n" + "="*70)
    print("NEW REQUEST")
    print("="*70)
    
    try:
        job_input = event["input"]
        
        # Validate input
        if "base64_pdf" not in job_input:
            return {
                "success": False,
                "data": None,
                "error": "Missing 'base64_pdf' in input"
            }
        
        # Get parameters
        dpi = job_input.get("dpi", 200)
        return_markdown = job_input.get("return_markdown", False)
        
        # Step 1: Decode PDF
        try:
            pdf_bytes = base64.b64decode(job_input["base64_pdf"])
            print(f"[1/4] PDF decoded ({len(pdf_bytes)} bytes)")
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Invalid base64: {str(e)}"
            }
        
        # Step 2: Convert to images
        try:
            images = pdf_to_images(pdf_bytes, dpi=dpi)
            print(f"[2/4] Converted to {len(images)} image(s) (DPI: {dpi})")
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"PDF conversion failed: {str(e)}"
            }
        
        # Step 3: DeepSeek OCR
        try:
            markdown_text = extract_markdown_from_images(images, output_path="/tmp/")
            print(f"[3/4] OCR completed ({len(markdown_text)} chars)")
            
            if not markdown_text or len(markdown_text) < 10:
                print("[WARNING] Markdown is empty or too short!")
                return {
                    "success": False,
                    "data": None,
                    "error": "OCR produced no output"
                }
                
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                "success": False,
                "data": None,
                "error": f"OCR failed: {str(e)}"
            }
        
        # Step 4: Gemini JSON extraction
        try:
            extracted_data = extract_json_with_gemini(markdown_text)
            print(f"[4/4] Gemini extraction completed")
            
            # Check if extraction failed
            if "error" in extracted_data:
                print(f"[WARNING] Gemini returned error: {extracted_data['error']}")
                # Still return the response, but mark as partial failure
                return {
                    "success": False,
                    "data": extracted_data,
                    "error": "Gemini extraction failed",
                    "markdown": markdown_text if return_markdown else None
                }
                
        except Exception as e:
            print(f"[ERROR] Gemini failed: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                "success": False,
                "data": None,
                "error": f"Gemini failed: {str(e)}",
                "markdown": markdown_text if return_markdown else None
            }
        
        # Prepare successful response
        response = {
            "success": True,
            "data": extracted_data,
            "error": None
        }
        
        if return_markdown:
            response["markdown"] = markdown_text
        
        print("="*70)
        print("✓ SUCCESS")
        print("="*70 + "\n")
        
        return response
        
    except Exception as e:
        print(f"[CRITICAL ERROR] {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            "success": False,
            "data": None,
            "error": f"Critical error: {str(e)}"
        }


# ============================================================================
# START RUNPOD SERVERLESS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING RUNPOD SERVERLESS HANDLER")
    print("="*70 + "\n")
    runpod.serverless.start({"handler": handler})