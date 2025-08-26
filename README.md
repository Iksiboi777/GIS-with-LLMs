# AI Chatbot for LUCAS Geospatial Data Analysis (Gemini-Powered)

This repository contains the code for a project exploring the use of Google's Generative AI to create a chatbot capable of translating natural language queries into executable Python code for geospatial (GIS) analysis.

This project implements a sophisticated **plan-and-execute** architecture, where the Large Language Model (LLM) first acts as a "planner" to decompose the user's request into logical steps, and then as a "coder" to implement each step sequentially.

## The Core Problem

Inspired by the academic challenges of Natural Language-to-Code (NL-to-Code) translation, this project aims to bridge the gap between non-technical user requests (e.g., "show me all woodland areas in Croatia") and the complex `geopandas` script required to query and visualize the LUCAS land use dataset.

## System Architecture

The chatbot operates on a multi-stage pipeline powered by the **Google Generative AI (Gemini Pro)** API to ensure robust and context-aware code generation:

1.  **Planning Stage (The "Planner"):** The user's query is first sent to the Gemini Pro model with a prompt designed to break the problem down into a series of logical, sequential steps. This "Chain of Thought" approach creates a clear execution plan before any code is written.

2.  **Implementation Stage (The "Coder"):** The system then iterates through the generated plan. Each individual step is sent back to Gemini Pro with a second, highly-focused prompt to generate the specific Python code snippet for that task alone.

3.  **Execution & Self-Correction:** The generated code for each step is executed. If it fails, an **autonomous code-fixing loop** captures the error traceback and re-prompts Gemini Pro to debug and fix the script, creating a resilient, self-correcting system.

## Key Features

-   **Plan-and-Execute Architecture:** Implements a two-step agent-like system where one LLM call creates a plan and subsequent calls execute it, improving reliability over single-shot generation.
-   **Chain of Thought Prompting:** Leverages advanced prompt engineering to have the LLM "think step-by-step," leading to more logical task decomposition.
-   **Google Generative AI Integration:** Utilizes the **Gemini Pro** model via its Python SDK for both the planning and coding stages.
-   **Autonomous Code Debugging:** A feedback loop that uses LLM-generated error messages to correct faulty code.

## Tech Stack

-   **Generative AI:** Google Generative AI API (Gemini Pro), Prompt Engineering
-   **Python Libraries:** Geopandas, Matplotlib, Scikit-learn, NumPy, Pandas

## Notebooks

The `/notebooks` directory contains the exploratory and prototyping work for this project.

* `eda_lucas.ipynb` & `europe_eda.ipynb`: These notebooks contain the comprehensive **Exploratory Data Analysis (EDA)** of the LUCAS dataset. The process includes loading the raw survey data with `pandas`, cleaning it, and converting it into a `geopandas` GeoDataFrame. Key visualizations were created to plot the georeferenced survey points onto a map of Europe, providing critical insights into the data's geographic distribution and land cover characteristics that informed the chatbot's development.

* `draft.ipynb`: This is a development notebook used for **prototyping the core "plan-and-execute" logic**. It contains initial experiments and tests for making calls to the Google Generative AI (Gemini Pro) API and refining the prompts used in the final application scripts.

## Repository Structure

The project is organized using a standard Python package structure for clarity and scalability.
