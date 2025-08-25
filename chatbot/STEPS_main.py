import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoProcessor
import pandas as pd
import numpy as np
import geopandas as gpd
import logging
import os
import sys
from query_classifier import QueryClassifier
from dataset_metadata import COLUMN_METADATA, DATASET_WARNINGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.getLogger('transformers').setLevel(logging.ERROR)


# Example tensor operation check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Code is executing on device: {device}")


# Add these functions
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_dataset_context(query_type: str, ) -> str:
    context = []
    
    # Add column metadata
    context.append("\n## Available Columns:")
    for col, meta in COLUMN_METADATA.items():
        if (query_type == "geo" and "shapefile" in meta["dataset"]) or \
           (query_type != "geo" and "csv" in meta["dataset"]):
            entry = f"- {col}: {meta['description']} ({meta['dtype']})"
            if "missing" in meta:
                entry += f" | Missing: {meta['missing']}%"
            if "warning" in meta:
                entry += f" | WARNING: {meta['warning']}"
            context.append(entry)
    
    # Add dataset warnings
    rules = ["\n## Critical Rules:"]
    warnings = DATASET_WARNINGS["shapefile" if query_type == "geo" else "csv"]
    for idx, warning in enumerate(warnings, 1):
        rules.append(f"{idx}. {warning}")
    
    return "\n".join(context), "\n".join(rules)


def main():
        
    # model_name, kode = "mistralai/Mistral-7B-Instruct-v0.2", mistral2
    # model_name, kode = "HuggingFaceH4/zephyr-7b-beta", zephyr
    # model_name, kode = "teknium/OpenHermes-2.5-Mistral-7B", openhermes
    # model_name, kode = "upstage/SOLAR-10.7B-Instruct-v1.0", solar
    # model_name, kode = "mistralai/Mistral-7B-Instruct-v0.3", mistral3
    # model_name, kode = "meta-llama/Meta-Llama-3-8B-Instruct", llama
    # model_name, kode = "gradientai/Llama-3-8B-Instruct-Gradient-1048k", "gradient"
    model_name, kode = "bigcode/starcoder2-15b-instruct-v0.1", "starcoder"


    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device=0
    # )
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16, 
        device=0,   
    )
    
    # Initialize classifier
    classifier = QueryClassifier(pipe)
    
    # Load datasets
    csv_path = "/home/ikuseta/diplomski_projekt/autonomousGIS/STEPS/temp_lucas.csv"
    shapefile_path = "/home/ikuseta/diplomski_projekt/autonomousGIS/geo_dataframe/geo_dataframe.shp"
    coastline_path = "/home/ikuseta/diplomski_projekt/autonomousGIS/Europe/Europe_coastline.shp"
    
    df = pd.read_csv(csv_path)
    gdf = gpd.read_file(shapefile_path)
    coastline = gpd.read_file(coastline_path)
    
    # Sample query
    # user_query = "Which are the top 5 countries in Europe that have the most extractable aluminum, and how much do they have?" THIS ONE IS GOOD, but needs more
    user_query = "Show me whether I can be certain that wherever I look for potassium, I will find calcium carbonate as well. Highlight the regions in Europe in which that is true." 

    query_type = classifier.classify(user_query)
    
    if query_type == 'geo':
        # Prepare context
        file_paths = [
            f"""You are working with a GeoDataFrame that is located in '/home/ikuseta/diplomski_projekt/autonomousGIS/geo_dataframe/geo_dataframe.shp'.
                      \n\nThese are the columns of the dataframe:{gdf.columns}\n\nThis is the head of the dataframe:{gdf.head()}
\n\nPlot the Europe shapefile that is located in '/home/ikuseta/diplomski_projekt/autonomousGIS/Europe/Europe_coastline.shp'.
\n\nThese are the columns of the Europe Shapefile:{coastline.columns}\n"""
        ]
    else:
        file_paths = [
            f"""You are working with a dataframe that is located in '{csv_path}'.
            \n\nThese are the columns of the dataframe:{df.columns}\n\nThis is the head of the dataframe:{df.head()}\n
            """
        ]
        # Prepare context
    file_paths += [get_dataset_context(query_type)[0]]
    print(file_paths[0])
    rules = [get_dataset_context(query_type)[1]]
    print(rules[0])
    
    # Configure outputs
    code_dir = "/home/ikuseta/diplomski_projekt/autonomousGIS/correct_code"
    output_dir = "/home/ikuseta/diplomski_projekt/autonomousGIS/code_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "response.txt")
    code_path = os.path.join(code_dir, "generated_code.py")
    
    # Generate steps
    from STEPS_generation import generate_steps
    print("############# STEP GENERATION #############")
    steps = generate_steps(pipe, file_paths, user_query, [], rules)
    
    # Implement code
    from STEPS_implementation import implement_steps
    print("############# CODE IMPLEMENTATION #############")
    code = implement_steps(pipe, file_paths, user_query, steps, [], rules)
    
    # Save results
    with open(output_path, "w") as f1, open(code_path, "w") as f2:
        f1.write(f"Query: {user_query}\n\n")
        f1.write(f"Classification: {query_type}\n\n")
        f1.write("Generated Steps:\n")
        f1.write(steps + "\n\n")
        f1.write("Generated Code:\n" + code)
        f2.write(code)
    
    clear_gpu_memory()

if __name__ == "__main__":
    main()



































# file_paths = [f"You are working with a GeoDataFrame that is located in '/home/fkriskov/diplomski/datasets/geo_dataframe.shp'.\n\nThese are the columns of the dataframe:{columns}\n\nThis is the head of the dataframe:{df_str}\n",
# f"Plot the Europe shapefile that is located in '/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp'.\n\nThese are the columns of the Europe Shapefile:{europe_columns}\n"]
# europe = gpd.read_file("/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp")


# "You are working with a GeoDataFrame that is located in '/home/fkriskov/diplomski/datasets/geo_dataframe.shp'."
# "Plot the Europe shapefile that is located in '/home/fkriskov/diplomski/datasets/Europe_coastline_shapefile/Europe_coastline.shp'."

# Descriptive statistics
# obj, filename = "Which land type (LC₀_Desc) has the highest pH_H₂O?", "desc_1_"
# obj, filename = "Plot the average 'OC' for each land type (LC₀_Desc). save it as a png.", "desc_2_"
# obj, filename = "Calculate the average pH for south EU.", "desc_3_"
# obj, filename = "Calculate the average pH for Austria, from the mentioned csv.", "desc_4_"
# obj, filename = "Calculate the max value of 'N' for Slovenia, from the mentioned csv.", "desc_5_"
# obj, filename = "Calculate the summary statistics for all numerical columns in the dataset.", "desc_6_"
# obj, filename = "Generate a correlation matrix of these columns: EC, pH_CaCl₂, pH_H₂O, OC, CaCO₃, P, N, K and visualize it using a heatmap.", "desc_7_"
# obj, filename = "Plot the distribution of 'K' with a KDE overlay. save it as a png.", "desc_8_"
# obj, filename = "Calculate the average 'K' for rows where 'EC' is greater than 10.", "desc_9_"
# obj, filename = "Find the sum of 'K' for each unique value in the 'LC₀_Desc' column. print the result.", "desc_10_"


# Inferential statistics

# obj, filename = "Is there a significant relationship between land type (LC₀_Desc) and pH_H₂O? Use chi square from scipy.", "infer_1_" 
# obj, filename = "Is there a significant difference between 'N' in Austria and France? Use ANOVA from scipy.", "infer_2_"
# obj, filename = "Which parameter has the strongest correlation with EC among {pH_CaCl₂, pH_H₂O, OC, CaCO₃, P, N, K}?", "infer_3_"
# obj, filename = "Perform a t-test to compare 'K' between Grassland and Cropland.", "infer_4_"
# obj, filename = "Plot a linear regression analysis to see the relationship between 'pH_H₂O' and 'K'. save it as a png", "infer_5_"
# obj, filename = "Construct a 95% \confidence interval for the mean 'OC' content in the dataset.", "infer_6_"
# obj, filename = "Using the Central Limit Theorem, simulate the sampling distribution of the mean 'pH_H₂O' from the dataset for sample sizes of 30. Plot the distribution and compare it to the normal distribution.", "infer_7_"
# obj, filename = "Calculate the z-scores for 'EC' and identify any outliers (z-score > 3 or < -3). use zscore from scipy", "infer_8_"
# obj, filename = "Perform a hypothesis test to determine if the mean 'K' content in the entire dataset is significantly different from 2%. Use a t-test for the hypothesis test.", "infer_9_"
# obj, filename = "Calculate and print out the p-value for the correlation between 'P' and 'K'. Determine if the correlation is statistically significant. Use pearsonr from scipy", "infer_10_"

# Geo-information

# obj, filename = "Plot all the points that have pH_CaCl₂ > 6. Use geopandas to plot the points and matplotlib to save the image as a png.", "geo_1_"
# obj, filename = "Plot all the points with LC₀_Desc=Woodland in Europe. Save the result as a png. Use geopandas.", "geo_2_"
# obj, filename = "Plot all the points with LC₀_Desc=Woodland & pH_H₂O <6. Save the result as a png. Use geopandas.", "geo_3_"
# obj, filename = "Perform sklearn.cluster.KMeans clustering on the TH_LAT and TH_LONG data to identify 3 clusters and plot them on a map. Use save it as a png.", "geo_4_"
# obj, filename = "Create a map with markers for all locations where 'K' is above its median value, in Europe. use geopandas. save the result as a png.", "geo_5_"
# obj, filename = "Generate a heatmap where each point is weighted by 'pH_CaCl₂'. Don't merge these shapefiles just plot them. Use geopandas. save the result as a png.", "geo_6_"
# obj, filename = "Create a map with markers for points where 'K' is in the top 10 percentile, in Europe. Don't merge these shapefiles just plot them. use geopandas. save the result as a png.", "geo_7_"
# obj, filename = "Plot the points with 'pH_H₂O'>5 blue and 'pH_H₂O'<5 red in Europe. save it as a png", "geo_8_"
# obj, filename = "Create a map displaying the distribution of soil types ('LC₀_Desc') across Europe. Each soil type should be represented by a different color. Use geopandas and save the map as a png.", "geo_9_"
# obj, filename = "Plot all the LC₀_Desc='Grassland' and LC₀_Desc='Woodland' points where 'OC'>20. Use geopandas and save the map as a png.", "geo_10_"
# obj, filename = "Show me all the cities in which they are very high levels of potassium", "geo_11_"


# # Add additional file_paths if plotting Geo-information
# if "geo" in filename:
#     file_paths=[f"""You are working with a GeoDataFrame that is located in '/home/ikuseta/diplomski_projekt/autonomousGIS/geo_dataframe/geo_dataframe.shp'.
#                       \n\nThese are the columns of the dataframe:{columns}\n\nThis is the head of the dataframe:{df_str}\n
# Plot the Europe shapefile that is located in '/home/ikuseta/diplomski_projekt/autonomousGIS/Europe/Europe_coastline.shp'.
# \n\nThese are the columns of the Europe Shapefile:{europe_columns}\n

# Notes: Always use matplotlib.pyplot to save the image as a png.
# \n\nAlways use a marker type '.' and size 5 to plot points. 
# \n\nAlways color the coastline lightgrey. 
# \n\nPlease, do not create functions, just a set of commands to do the task.
# \n\nDo not merge the shapefiles, under any circumstance.
# """]
    
# else:
#     file_paths = [f"""You are working with a CSV file that is located in '/home/ikuseta/diplomski_projekt/autonomousGIS/STEPS/temp_lucas.csv'.
# \n\nThese are the columns of the dataframe:{columns}\n\nThis is the head of the dataframe:{df_str}\n

# Note: The column name 'LC₀_Desc' contains a subscript zero (₀), not a regular zero (0).
# If a prompt is not precise enough when it comes to regions, ask the user to elaborate further and go straight to User Loop.
# Please maintain this exact format in generated code.

# """]


# # First, set up the output directory
# output_dir = "/home/ikuseta/diplomski_projekt/autonomousGIS/evaluation"  # Modify this path as needed
# os.makedirs(output_dir, exist_ok=True)



# output_path = os.path.join(output_dir, kode, filename+kode+".txt")
# code_path = os.path.join("/home/ikuseta/diplomski_projekt/autonomousGIS/correct_code", filename+".py")
# results_path = os.path.join("/home/ikuseta/diplomski_projekt/autonomousGIS/code_results", filename)

# # Save the output
# with open(output_path, 'w', encoding='utf-8') as f_out, open(code_path, 'w', encoding='utf-8') as f_code:

#     print("############# STEP GENERATION #############")
#     steps = generate_steps(pipe, file_paths, obj, [])
#     f_out.write("############# STEP GENERATION #############\n")
#     f_out.write(steps + "\n\n")

#     print("############# CODE IMPLEMENTATION #############")
#     code = implement_steps(pipe, code_path, results_path, file_paths, obj, steps, [])
#     f_out.write("############# CODE IMPLEMENTATION #############\n")
#     f_out.write(code)
#     f_code.write(code)


# clear_gpu_memory()


# # User Loop
# while True:
#     print("############# USER LOOP #############")
#     oldmessages = [
#             {
#                 "role": "user",
#                 "content": f"""
#     "for these files: {"".join(file_paths)}"
#     "answers this user query:{obj}"
#     """
#             },
#             {
#                 "role": "assistant",
#                 "content": f"""
#     {steps}

#     {code}
#     """
#             },
#         ]

#     obj = input("Is there something else you want to ask? Ask away: ")
#     if obj in ["exit", "EXIT", "Exit", "e", "E", "No", "no"]:
#         #clear_gpu_memory()
#         break
       

#     print("############# STEP GENERATION #############")
#     steps = generate_steps(pipe, file_paths, obj, oldmessages)

#     with open(code_path, 'w', encoding='utf-8') as f_code, open(results_path, 'w', encoding='utf-8') as f_res:
#         print("############# CODE IMPLEMENTATION #############")
#         code = implement_steps(pipe, code_path, results_path, file_paths, obj, steps, oldmessages)
#         f_code.write(code)
#         f_res.write(code)