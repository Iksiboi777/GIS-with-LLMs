from error_code_fixer import error_code_fixer, double_check
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('transformers').setLevel(logging.ERROR)


def implement_steps(pipe, file_paths, user_query, steps, messages, rules):
    instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The solution should be a Python expression that can be called with the `exec()` function, inside '```python ```'.\n"
    "3. The code should represent a solution to the query.\n"
    "4. If not instructed otherwise, print the final result variable.\n"
    "5. If you are asked to plot something, save it as a .png.\n"
    "6. Don't explain the code.\n"
    )

# "You cannot merge these shapefiles,just plot them."
# "set marker='.' and figsize (10,10)"

    messages += [
        {
            "role": "user",
            "content": f"""
"files needed are {"".join(file_paths)}"

"'NUTS_0' is Alpha-2 code."

"for the given file locations and for these solution steps: {steps}"

"generate a complete python code that follows these steps and answers this user query:{user_query}"

"You must follow these rules: {rules}"

{instruction_str}
"""
# "The possible NUTS_0 codes are: {codes}"
        },
    ]

    invalid_output = True
    temp_messages = messages
    while invalid_output:
        prompt = pipe.tokenizer.apply_chat_template(temp_messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.4, top_k=50, top_p=0.95)
        result = outputs[0]["generated_text"]
        print(result)
        
        pythonbracketcounter = instruction_str.count('```python') + 1
        if result.count("```python") == pythonbracketcounter:
            # temp_messages = messages + ["Display only the complete Python solution.\n"]
            invalid_output = False

    python_extract = result.split("```python")[pythonbracketcounter].split("```")[0]
    print("\n------------------GREAT SUCCESS!!!------------------\n")
    print(python_extract)
    print("\n------------------REZULTAT!!!------------------\n")

    
    # double check
    checked_code = double_check(pipe, python_extract, messages)
    # print(checked_code)
    try:
        error_occured = False
        print("\nRunning Code...\n")
        exec(checked_code)
    except Exception as e:
        error_occured = True
        error = e
        print("\nerror occured: ", e, "\n")


    # error code fixer
    python_fix_extract = checked_code
    while error_occured:
        python_fix_extract = error_code_fixer(pipe, python_fix_extract, error)

        print("\n------------------FIXED!!!------------------\n")
        print(python_fix_extract)
        print("\n------------------REZULTAT!!!------------------\n")
        try:
            error_occured = False
            exec(python_fix_extract)
        except Exception as e:
            error_occured = True
            error = e
            print("\nerror occured: ", e, "\n")

    return python_fix_extract



















# import re
# from dataset_metadata import COLUMN_METADATA, DATASET_WARNINGS

# def validate_code(code: str, query_type: str) -> bool:
#     # Check for prohibited patterns
#     errors = []
    
#     if query_type == "geo":
#         if not re.search(r"plt\.savefig\(.*?\.png\)", code):
#             print("ERROR: Must save as PNG with matplotlib")
#             return False
            
#         if not re.search(r"plot\(.*?marker='\.'.*?markersize=5", code):
#             print("ERROR: Missing marker '.' size 5")
#             return False
            
#         if "coastline.plot(color='lightgrey')" not in code:
#             print("ERROR: Coastline not colored lightgrey")
#             return False
            
#         if "gpd.overlay" in code or "gpd.sjoin" in code:
#             print("ERROR: Shapefile merging detected")
#             return False
        
#     else:
#         if "geopandas" in code.lower():
#             errors.append("Illegal use of geopandas in non-geo query")
    
#     # Check for required handling
#     for col, meta in COLUMN_METADATA.items():
#         if meta.get("handling") and col in code:
#             if not re.search(meta["handling"].split("(")[0], code):
#                 errors.append(f"Missing handling for {col}: {meta['handling']}")
    
#     if errors:
#         print(f"Validation failed: {', '.join(errors)}")
#         return False
#     return True


# def implement_steps(pipe, code_path, results_path, file_paths, user_query, steps, messages, query_type):
#     full_prompt = f"""Generate Python code for this agricultural analysis:
    
#     Query: {user_query}
#     Type: {query_type.upper()}
#     Required Columns: {', '.join(COLUMN_METADATA.keys())}
    
#     Requirements:
#     1. {"Use geopandas and shapefiles" if query_type == "geo" else "Use pandas and CSV"}
#     2. {DATASET_WARNINGS["shapefile" if query_type == "geo" else "csv"][0]}
#     3. Save {"plot" if 'plot' in user_query.lower() else "results"} to {results_path}
    
#     Steps to implement:
#     {steps}
    
#     Code Template:
#     ```python
#     # Imports
#     {"import geopandas as gpd" if query_type == "geo" else "import pandas as pd"}
#     import matplotlib.pyplot as plt
    
#     # Data loading
#     {"gdf = gpd.read_file(...)" if query_type == "geo" else "df = pd.read_csv(...)"}
    
#     # Analysis steps
#     ...
    
#     # Output
#     plt.savefig(...)  # For plots
#     print(...)        # For textual results
#     ```"""
    
#     # Generate code
#     messages += [
#         {"role": "user", "content": full_prompt}
#     ]
#     prompt = pipe.tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     outputs = pipe(
#         prompt,
#         max_new_tokens=1024,
#         do_sample=True,
#         temperature=0.3,
#         top_k=50,
#         top_p=0.95
#     )
    
#     # Extract code
#     raw_code = outputs[0]["generated_text"]
#     code_blocks = re.findall(r"```python(.*?)```", raw_code, re.DOTALL)
#     final_code = "\n".join([cb.strip() for cb in code_blocks])
#     print("FInal result: ", final_code)
    
#     # Validate and retry
#     if not validate_code(final_code, query_type):
#         return implement_steps(pipe, code_path, results_path, file_paths, 
#                               user_query, steps, messages + [{"role": "user", "content": "Fix previous errors"}], query_type)
    
#     # Save and return
#     with open(code_path, "w") as f:
#         f.write(final_code)
    
#     return final_code