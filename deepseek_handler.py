import runpod
import os
import json
import base64
import tempfile
import torch
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import AutoModel, AutoTokenizer
import google.generativeai as genai

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

print("Initializing models...")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True
).eval()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))

HANDLER_VERSION = os.environ.get("HANDLER_VERSION", "handler-2025-10-30-01")

print(f"Models ready (version: {HANDLER_VERSION})\n")

# ============================================================================
# FUNCTIONS
# ============================================================================

def pdf_to_images(pdf_bytes, dpi=200):
    images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt='png')
    return [img.convert("RGB") for img in images]


def extract_markdown_from_images(images):
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
    
    combined = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for im in resized:
        combined.paste(im, (0, y))
        y += im.height
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        combined_path = tmp.name
        combined.save(combined_path, quality=95)
    
    try:
        # Primary approach with explicit output_path to avoid mkdir('') issues
        res = model.infer(
            tokenizer,
            prompt="<image>\nConvert to markdown.",
            image_file=combined_path,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            test_compress=True,
            eval_mode=True,
            save_results=False,
            output_path=tempfile.gettempdir(),
        )
        markdown_text = str(res).strip()
        
    except Exception as e:
        # Fallback: Use tempfile directory
        print(f"Primary approach failed: {e}")
        print("Using tempfile fallback...")
        
        temp_dir = tempfile.gettempdir()
        res = model.infer(
            tokenizer,
            prompt="<image>\n<|grounding|>Convert the document to markdown.",
            image_file=combined_path,
            output_path=temp_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=True,
            eval_mode=True
        )
        markdown_text = str(res).strip()
    
    finally:
        try:
            os.remove(combined_path)
        except:
            pass
    
    return markdown_text


def extract_json_with_gemini(markdown_text):
    if len(markdown_text) > 100000:
        markdown_text = markdown_text[:100000]
    
    prompt = f"""
You are an expert load confirmation analyst with over 10 years of experience as a dispatcher and broker in the trucking industry. You are thoroughly familiar with all possible terminologies used in load confirmation, dispatch, and rate agreement documents—including synonyms and alternate phrasing (for instance, "commodity" can also appear as "goods" or "product"; "carrier pay" as "freight rate" or "total payment"; etc.). Your role is to ensure that not a single piece of critical information is missed, regardless of wording, layout variation, or format.

Your task is:

Given the following Markdown-formatted text extracted from a load confirmation, rate confirmation, or dispatch document:

Identify and extract every key data point required for carrier load execution and payment, even if fields are phrased differently or appear in varied order.

Ensure you cross-reference synonymous terms and do not overlook information if it appears under a different label.

Omit no critical details that would impact payment, scheduling, claims, compliance, or operational clarity.

Strictly avoid inventing or hallucinating values. If a field is not present in the document, output an empty string, array, or object as appropriate.
tip : Have total grasp of the whole extracted data, so try to understand what is what and correctly fit in the JSON structure even the text is far away from.

Present the output exactly in the following JSON structure (and nothing else—no commentary, explanation, or extra text):

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

Fill each field only if the information is explicitly listed, regardless of the label or location.

Leave fields blank (empty string, array, or object) if not present.

Do not output any extra text—only the JSON as specified above.

Prioritize accuracy and completeness, accounting for all synonym variations and industry-specific language.

This will ensure consistent, comprehensive, and error-free extraction of critical load information across any truckload document.

OCR MARKDOWN:
{markdown_text}

"""
    
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=8092,
            response_mime_type="application/json",
        ),
        safety_settings=safety_settings
    )
    
    if not response.candidates:
        return {"error": "Response blocked by safety filters"}
    
    candidate = response.candidates[0]
    
    try:
        text = response.text.strip()
    except ValueError:
        if candidate.content and candidate.content.parts:
            text = candidate.content.parts[0].text.strip()
        else:
            return {"error": "No text in response"}
    
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {str(e)}", "raw_response": text[:1000]}


# ============================================================================
# HANDLER
# ============================================================================

def handler(event):
    print("\n" + "="*70)
    print("NEW REQUEST")
    print("="*70)
    
    try:
        job_input = event["input"]
        
        if "base64_pdf" not in job_input:
            return {"success": False, "data": None, "error": "Missing 'base64_pdf'"}
        
        dpi = job_input.get("dpi", 200)
        return_markdown = job_input.get("return_markdown", False)
        
        pdf_bytes = base64.b64decode(job_input["base64_pdf"])
        print(f"[1/4] PDF decoded: {len(pdf_bytes)} bytes")
        
        images = pdf_to_images(pdf_bytes, dpi=dpi)
        print(f"[2/4] Converted to {len(images)} images")
        
        markdown_text = extract_markdown_from_images(images)
        print(f"[3/4] OCR complete: {len(markdown_text)} chars")
        
        # Print markdown for debugging
        print("\n" + "="*70)
        print("EXTRACTED MARKDOWN")
        print("="*70)
        print(markdown_text)
        print("="*70 + "\n")
        
        if len(markdown_text) < 10:
            return {"success": False, "data": None, "error": "OCR output too short"}
        
        extracted_data = extract_json_with_gemini(markdown_text)
        print(f"[4/4] Extraction complete")
        print(extracted_data)
        if "error" in extracted_data:
            return {
                "success": False,
                "data": extracted_data,
                "error": "Extraction failed",
                "markdown": markdown_text if return_markdown else None
            }
        
        response = {
            "success": True,
            "data": extracted_data,
            "error": None,
            "version": HANDLER_VERSION
        }
        
        if return_markdown:
            response["markdown"] = markdown_text
        
        print("SUCCESS\n")
        return response
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"success": False, "data": None, "error": str(e)}


# ============================================================================
# START
# ============================================================================

if __name__ == "__main__":
    print("\nSTARTING RUNPOD HANDLER\n")
    runpod.serverless.start({"handler": handler})