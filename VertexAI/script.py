import pandas as pd
from tqdm.auto import tqdm
from google import genai
from google.genai import types
from pathlib import Path
import json
import time


from utils import building_prompt, building_json_response, download_codebook, download_examples, download_data, build_repair_instruction_prompt
from secret import project_ID


base_dir = Path.cwd()
generations_dir = base_dir / "generations"
scl_binary = ["DIFFINST", "CWCM", "MQI_CHECK", "ORIENT", "SUMM"]
scl_three = ["APLPROB", "DIRINST", "ETCA", "EXPL", "LANGIMP", "LCP", "LINK", "LLC", "MAJERR", "MGEN", "MLANG", "MMETH", "MQI3", "OERR", "ORICH", "OWWS", "REMED", "SMQR", "STEXPL", "USEPROD", "WCDISS"]
scl_four = ["MATCON", "MMSM", "OERR4", "ORICH4", "OSPMMR4", "OWWS4", "SMALDIS", "STUCON"]
scl_five = ["LESSEFFIC", "MQI5", "TASKDEVMAT", "TSTUDEA"] 

# ================== TO ADJUST ==================
# ===== CONFIGURATIONS =====
# codebook_type = "reasoning" # reasoning, no-reasoning, simplified, simplest, no-scale
# examples_type = "reasoning" # reasoning, no-reasoning, none
### NO THINKING - NOT for 2.5-pro
# combination = [("reasoning", "reasoning", "reasoning_codebook_examples-with-reasoning.csv"),
#                ("reasoning", "no-reasoning", "reasoning_codebook_examples-with-no-reasoning.csv"),
#                ("no-reasoning", "no-reasoning", "codebook_examples-with-no-reasoning.csv"),
#                ("no-reasoning", "none", "codebook.csv"),
#                ("simplified", "none", "simplified.csv"),
#                ("simplest", "none", "simplest.csv"),
#                ("no-scale", "none", "no-scale.csv")
#                ]
# thinking_credits = 0

### THINKING
# combination = [("no-reasoning", "reasoning", "thinking_codebook_examples-with-reasoning.csv"),
#                ("no-reasoning", "no-reasoning", "thinking_codebook_examples-with-no-reasoning.csv"),
#                ("no-reasoning", "none", "thinking_codebook.csv"),
#                ("simplified", "none", "thinking_simplified.csv"),
#                ("simplest", "none", "thinking_simplest.csv"),
#                ("no-scale", "none", "thinking_no-scale.csv")
#                ]
# thinking_credits = 256 # FOR 2.5 Flash and 2.5-pro
# thinking_credits = 512 # ONLY FOR Flash-lite
### BEST CONFUGURATION (reasoning + reasoning examples)
combination = [("reasoning", "reasoning", "reas_ex-w-reas.csv")]
thinking_credits = 0

### ===== VARIABLES =====
all_variables = sorted(
    scl_binary + scl_three + scl_four + scl_five
)
# all_variables = ["STUCON"]
# all_variables = ["LCP"]
# all_variables = ["MLANG"]

### INPUT DIRECTORY
# datasets_dir = base_dir / "data" / "balanced_subsets"
datasets_dir = base_dir / "data" / "teacher_subsets"

### ===== OUTPUT DIRECTORY =====
#### ALL
# output_dir = base_dir / "VertexAI" / "2.5-flash-lite_short-reasoning_examples"
# output_dir = base_dir / "VertexAI" / "2.5-flash_best-config"
output_dir = base_dir / "VertexAI" / "teachers_2.5-flash_best-config"
#### STUCON
# output_dir = base_dir / "VertexAI" / "STUCON_2.5-flash-lite_tests"
# output_dir = base_dir / "VertexAI" / "STUCON_2.5-flash_tests"
# output_dir = base_dir / "VertexAI" / "STUCON_2.5-pro_tests"
#### LCP
# output_dir = base_dir / "VertexAI" / "LCP_2.5-flash-lite_tests"
# output_dir = base_dir / "VertexAI" / "LCP_2.5-flash_tests"
# output_dir = base_dir / "VertexAI" / "LCP_2.5-pro_tests"
#### MLANG
# output_dir = base_dir / "VertexAI" / "MLANG_2.5-flash-lite_tests"
# output_dir = base_dir / "VertexAI" / "MLANG_2.5-flash_tests"
# output_dir = base_dir / "VertexAI" / "MLANG_2.5-pro_tests"

# ===== MODEL AND COSTS =====
# Models
# MODEL = "gemini-2.5-flash-lite"
# input_cost = 0.10
# output_cost = 0.40
MODEL = "gemini-2.5-flash"
input_cost = 0.30
output_cost = 2.5
# MODEL = "gemini-2.5-pro"
# input_cost = 1.25
# output_cost = 10
# MODEL = "gemini-3-flash-preview" # NOT USED

temperature = 0
max_tokens = 500

system_instruction = "You are a strict and precise annotator who avoids any unnecessary leniency."

# ===================================================

# Vertex AI settings
LOCATION = "europe-west1"
# Create the Vertex AI client
client = genai.Client(
    vertexai=True,
    project=project_ID(),
    location=LOCATION
)

total_cost_usd = 0
for comb in combination:
    codebook_type, examples_type, output_filename = comb
    for turn in range(1):
        for var in all_variables:
            # Price
            total_in = 0
            total_out = 0

            prep_folder_path = generations_dir / var
            # Downloading codebook
            codebook = download_codebook(prep_folder_path, codebook_type)
            # Downloading examples
            examples = ""
            if examples_type != "none":
                examples = download_examples(prep_folder_path, examples_type)

            # Building json response schema    
            allowed_values, response_json_schema = building_json_response(var, codebook_type)
            
            # Data
            df = download_data(datasets_dir, var)

            if codebook_type == "reasoning":
                reasonings = []
            ratings = []
            for text in tqdm(df["full_text"], total=len(df), desc=f"Generating {var}"): # - {output_filename}"):
                # Building prompt
                prompt = building_prompt(codebook, examples, text, codebook_type, examples_type)

                success = False
                attempt = 0
                max_attempts = 4
                temp_tokens = max_tokens
                base_delay = 0.3
                while not success and attempt < max_attempts:
                    temp_tokens = int(max_tokens + 200 * attempt)

                    attempt += 1
                    try:
                        response = client.models.generate_content(
                            model=MODEL,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(thinking_budget=thinking_credits),
                                temperature=temperature,
                                max_output_tokens=temp_tokens,
                                system_instruction=system_instruction,
                                response_mime_type="application/json",
                                response_schema=response_json_schema,
                            ),
                        )
                        u = response.usage_metadata
                        in_tokens = u.prompt_token_count or 0
                        total_in += in_tokens
                        out_tokens = u.candidates_token_count or 0
                        thinking_tokens = u.thoughts_token_count or 0
                        total_out += out_tokens + thinking_tokens                    
                        
                        # json parsing
                        try:
                            obj = json.loads(response.text)

                            if codebook_type == "reasoning":
                                if ("your_reasoning" not in obj) or ("rating" not in obj):
                                    raise ValueError("Missing keys after repair")
                                reasonings.append(obj.get("your_reasoning"))
                            else:
                                if ("rating" not in obj):
                                    raise ValueError("Missing keys after repair")

                            ratings.append(obj.get("rating"))
                            success = True
                            break

                        except Exception as e:
                            tqdm.write(f"[{attempt}/{max_attempts}] Bad model output: {e}")
                            tqdm.write(response.text[-200:])

                            # Try a single "JSON repair" call (cheaper than full regeneration)
                            try:
                                repair_instruction, repair_prompt = build_repair_instruction_prompt(response.text)

                                repair_response = client.models.generate_content(
                                    model="gemini-2.5-flash-lite",
                                    contents=repair_prompt,
                                    config=types.GenerateContentConfig(
                                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                                        temperature=0.0,
                                        max_output_tokens=500,
                                        system_instruction=repair_instruction,
                                        response_mime_type="application/json",
                                        response_schema=response_json_schema,
                                    ),
                                )
                                obj = json.loads(repair_response.text)

                                if codebook_type == "reasoning":
                                    if ("your_reasoning" not in obj) or ("rating" not in obj):
                                        raise ValueError("Missing keys after repair")
                                    reasonings.append(obj.get("your_reasoning"))
                                else:
                                    if ("rating" not in obj):
                                        raise ValueError("Missing keys after repair")

                                ratings.append(obj.get("rating"))
                                tqdm.write("Successfull repair!")
                                success = True
                                break

                            except Exception as repair_e:
                                tqdm.write(f"[{attempt}/{max_attempts}] JSON repair failed: {repair_e}")
                                continue

                    except Exception as e:
                        msg = str(e).lower()

                        # Retry only on specific transient errors inferred from the message
                        transient = (
                            "overloaded" in msg
                            or "resource exhausted" in msg
                            or "quota" in msg
                            or "429" in msg
                            or "unavailable" in msg
                            or "try again" in msg
                        )

                        # Non-transient error
                        if not transient:
                            tqdm.write(f"[{attempt}/{max_attempts}] Non-transient error, skipping row: {e}")
                            # time.sleep(3*60) # If connection issues
                            break

                        # Exponential backoff between retries
                        delay = base_delay * (1.5 ** (attempt - 1))
                        tqdm.write(f"[Gemini] Transient error (attempt {attempt}/{max_attempts}): {e}")
                        tqdm.write(f"[Gemini] Retrying in {delay:.2f}s...")
                        time.sleep(delay)

                if not success:
                    if codebook_type == "reasoning":
                        reasonings.append(None)
                    ratings.append(None)
 
            if codebook_type == "reasoning":
                df["your_reasoning"] = reasonings
            df["rating"] = ratings

            # CSV Export
            # output_file_name = output_dir / output_filename
            output_file_name = output_dir / (var + "_" + output_filename)
            df.to_csv(output_file_name, index=False, encoding="utf-8")

            # Cost calculation
            cost_usd = (total_in / 1_000_000) * input_cost + (total_out / 1_000_000) * output_cost
            print("input_tokens:", total_in, "output_tokens:", total_out, "estimated_cost_usd:", cost_usd)
            total_cost_usd += cost_usd

print("total cost in usd:", total_cost_usd)