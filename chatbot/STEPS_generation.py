

def generate_steps(pipe, file_paths, obj, messages, rules):
    # StarCoder2 requires code-style prompting, not chat templates
    system_prompt = f"""# Agricultural Analysis Step Generation\n
    For the given objective: {obj}\nand the available data: {file_paths[0]}
    \nCome up with a step by step plan with no more than 5-7 steps. This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
    \nSteps should be clearly noted by having 'Step X:' written before the step itself, where X is the step number. \
    DO NOT WRITE ANY CODE!!!!!
    These are rules you have to abide by:\n{rules[0]}
"""

    messages += [
        {"role": "user", "content": system_prompt}
    ]
    
    # Generate response
    prompt = pipe.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95
    )
    
    # Process output
    raw_result = outputs[0]["generated_text"]

    finalres = ""
    remove_python = raw_result.split("```python")
    for each in remove_python:
        splited = each.split("```")
        each = splited[len(splited)-1]
        finalres += each

    # Additional cleanup to prevent repetition
    steps = [step.strip() for step in finalres.split('Step') if step.strip()]
    unique_steps = []
    seen = set()
    
    for step in steps:
        if step not in seen:
            seen.add(step)
            unique_steps.append(f"Step {step}")

    final_output = "\n".join(unique_steps)
    print("Final response:\n" + final_output)
    return final_output



# We'll see whether we will even have repetition problems later, but this is good











# from dataset_metadata import DATASET_WARNINGS

# def generate_steps(pipe, file_paths, obj, messages):
    
#     messages += [
#         {
#             "role": "user",
#             "content": f"""For the given objective: {obj}
# and these given files: {"".join(file_paths)}
# come up with a concise step by step plan with no more than 5-7 steps. \
# This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
# The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps. \
# Steps should be clearly noted by having 'Step X:' written before the step itself, where X is the step number. \
# DO NOT WRITE ANY CODE!!!!!
# """
#         },
#     ]

#     prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
#     result = outputs[0]["generated_text"]
#     print("Result:\n" + result)

#     finalres = ""
#     remove_python = result.split("```python")
#     for each in remove_python:
#         splited = each.split("```")
#         each = splited[len(splited)-1]
#         finalres += each

#     # Additional cleanup to prevent repetition
#     steps = [step.strip() for step in finalres.split('Step') if step.strip()]
#     unique_steps = []
#     seen = set()
    
#     for step in steps:
#         if step not in seen:
#             seen.add(step)
#             unique_steps.append(f"Step {step}")

#     final_output = "\n".join(unique_steps)
#     print("Final response:\n" + final_output)
#     return final_output
#     # print("Final response: " + finalres)
#     # return finalres
