import pandas as pd


def building_prompt(codebook, examples, text, codebook_type, examples_type):
    # 1/3 PROMPT with REASONING and EXAMPLES
    if codebook_type=="reasoning" and examples_type in ("reasoning", "no-reasoning"):
        return f"""
=== CODEBOOK ===
{codebook}

=== EXAMPLES ===
{examples}

=== INSTRUCTIONS ===
- Evaluate the next content according to the codebook and examples previously given.
- Your response must be a valid JSON object with exactly these keys:
  - "your_reasoning" must be single-line text, sentences separated by spaces only, no line breaks, no bullet points
  - "rating" must respect the values allowed
- Do not include any explanations, greetings, or additional text.

=== CONTENT TO ANNOTATE ===
<<<
{text}
>>>
"""

# 2/3 PROMPT no REASONING with EXAMPLES
    elif codebook_type != "reasoning" and examples_type in ("reasoning", "no-reasoning"):
        return f"""
=== CODEBOOK ===
{codebook}

=== EXAMPLES ===
{examples}

=== INSTRUCTIONS ===
- Evaluate the next content according to the codebook and examples previously given.
- Your response must be a valid JSON object with exactly this key:
  - "rating" must respect the values allowed
- Do not include any explanations, greetings, or additional text.

=== CONTENT TO ANNOTATE ===
<<<
{text}
>>>
"""

# 3/3 PROMPT no EXAMPLES no REASONING
    elif codebook_type != "reasoning" and examples_type == "none":
        return f"""
=== CODEBOOK ===
{codebook}

=== INSTRUCTIONS ===
- Evaluate the next content according to the codebook previously given.
- Your response must be a valid JSON object with exactly this key:
  - "rating" must respect the values allowed
- Do not include any explanations, greetings, or additional text.

=== CONTENT TO ANNOTATE ===
<<<
{text}
>>>
"""
    else:
        raise ValueError("Invalid combination of codebook_type/examples_type for prompt generation")



def building_json_response(var, codebook_type):
    scl_binary = ["DIFFINST", "CWCM", "MQI_CHECK", "ORIENT", "SUMM"]
    scl_three = ["APLPROB", "DIRINST", "ETCA", "EXPL", "LANGIMP", "LCP", "LINK", "LLC", "MAJERR", "MGEN", "MLANG", "MMETH", "MQI3", "OERR", "ORICH", "OWWS", "REMED", "SMQR", "STEXPL", "USEPROD", "WCDISS"]
    scl_four = ["MATCON", "MMSM", "OERR4", "ORICH4", "OSPMMR4", "OWWS4", "SMALDIS", "STUCON"]
    scl_five = ["LESSEFFIC", "MQI5", "TASKDEVMAT", "TSTUDEA"]

    if var in scl_binary:
        enum = [0, 1]
        enum_str = ["0", "1"]
    elif var in scl_three:
        enum = [1, 2, 3]
        enum_str = ["1", "2", "3"]
    elif var in scl_four:
        enum = [1, 2, 3, 4]
        enum_str = ["1", "2", "3", "4"]
    elif var in scl_five:
        enum = [1, 2, 3, 4, 5]
        enum_str = ["1", "2", "3", "4", "5"]
    else:
        enum = None
    
    rating_schema = {
        "type": "STRING",
        "nullable": False
    }

    if enum is not None:
        rating_schema["enum"] = enum_str

    # 1/2 JSON with REASONING
    if codebook_type=="reasoning":
        return enum, {
            "type": "OBJECT",
            "properties": {
                "your_reasoning": {
                    "type": "STRING",
                    "nullable": False
                },
                "rating": rating_schema
            },
            "required": ["your_reasoning", "rating"]
        }
    # 2/2 JSON no REASONING
    elif codebook_type in ("no-reasoning", "simplified", "simplest", "no-scale"):
        return enum, {
            "type": "OBJECT",
            "properties": {
                "rating": rating_schema
            },
            "required": ["rating"]
        }
    
    # Problems
    else:
        raise ValueError("Invalid codebook_type for JSON generation") 





def download_codebook(prep_folder_path, codebook_type):
    # 1/5 DOWNLOAD REASONING
    if codebook_type=="reasoning":
        codebook_path = prep_folder_path / "codebook preparation"

    # 2/5 DOWNLOAD no REASONING
    elif codebook_type=="no-reasoning":
        codebook_path = prep_folder_path / "codebook preparation_no-reasoning"
    
    # 3/5 DOWNLOAD SIMPLIFIED CODEBOOK
    elif codebook_type=="simplified":
        codebook_path = prep_folder_path / "codebook preparation_simplified"
    
    # 4/5 DOWNLOAD SIMPLEST CODEBOOK
    elif codebook_type=="simplest":
        codebook_path = prep_folder_path / "codebook preparation_simplest"
    
    # 5/5 DOWNLOAD no SCALE CODEBOOK
    elif codebook_type=="no-scale":
        codebook_path = prep_folder_path / "codebook preparation_no-scale"
    
    # Problems
    else:
        raise ValueError("Invalid codebook_type for codebook download")
    
    with open(codebook_path, "r", encoding="utf-8") as f:
        codebook = f.read()
    return codebook
    

def download_examples(prep_folder_path, examples_type):
    # 1/2 DOWNLOAD REASONING
    if examples_type=="reasoning":
        examples_path = prep_folder_path / "examples_LLM.txt"
    
    # 2/2 DOWNLOAD no REASONING
    elif examples_type in ("no-reasoning"):
        examples_path = prep_folder_path / "examples_LLM_no-reasoning.txt"
    
    # Problems
    else:
        raise ValueError("Invalid examples_type for examples download")
    
    with open(examples_path, "r", encoding="utf-8") as f:
            examples = f.read()
    return examples 






def download_data(datasets_dir, var):
    pre = "MQI_"
    post = "_"
    filename = pre + var + post
    file_path = next(
            f for f in datasets_dir.iterdir()
            if f.is_file() and f.name.startswith(filename)
    )
    df = pd.read_csv(file_path, dtype=str, encoding="utf-8")
    return df



def build_repair_instruction_prompt(bad_text: str) -> str:
    instruction = (
    "You are a strict JSON repair tool."
    "Return ONLY valid JSON that matches the provided schema."
    "Do not add extra keys. Do not include markdown or explanations."
    )
    prompt = (
        "Fix the following model output so that it is valid JSON and matches the required schema."
        "Return ONLY the corrected JSON.\n\n"
        "BAD_OUTPUT:\n"
        f"{bad_text}"
    )
    return instruction, prompt