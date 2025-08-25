# AI Chatbot for Geospatial Analysis (Plan-and-Execute LLM Pipeline)

This repository contains the code for an advanced "Chatbot for Farmers" project. It explores a sophisticated **plan-and-execute** architecture using Large Language Models (LLMs) to translate natural language queries into executable Python code for geospatial (GIS) analysis.

This project moves beyond simple single-shot code generation and implements a more robust, multi-stage pipeline where the LLM first acts as a "planner" to decompose the problem and then as a "coder" to implement each step.

## The Core Problem

Inspired by the challenges outlined in academic research on Natural Language-to-Code (NL-to-Code) translation, this project aims to bridge the gap between non-technical user requests (e.g., "show me crop cover change") and the complex, syntactically perfect code required to perform agricultural data analysis.

## System Architecture

The chatbot operates on a two-stage generative pipeline to ensure robust and context-aware code generation:

1.  **Planning Stage (The "Planner"):** The user's query is first sent to an LLM with a prompt designed to break the problem down into a series of logical, sequential steps. This "Chain of Thought" approach creates a clear execution plan before any code is written.

2.  **Implementation Stage (The "Coder"):** The system then iterates through the generated plan. Each individual step is sent to the LLM with a second, highly-focused prompt to generate the specific Python code snippet for that task alone.

3.  **Execution & Debugging:** The generated code for each step is executed. If it fails, the **autonomous code-fixing loop** captures the error and re-prompts the LLM to provide a corrected version, creating a resilient, self-correcting system.

## Key Features

-   **Plan-and-Execute Architecture:** Implements a two-step agent-like system where one LLM call creates a plan and subsequent calls execute it, improving reliability over single-shot generation.
-   **Chain of Thought Prompting:** Leverages advanced prompt engineering to force the LLM to "think step-by-step," leading to more logical and accurate task decomposition.
-   **Local LLM Integration:** Utilizes Ollama for local hosting of various open-source models (e.g., Qwen, Code Llama), enabling rapid and cost-effective experimentation.
-   **Context-Aware Generation:** Enriches prompts with conversation history and dataset metadata for more accurate code generation.

## Tech Stack

-   **Generative AI:** Ollama, LLMs (Qwen, Code Llama), Prompt Engineering
-   **Python Libraries:** Geopandas, Matplotlib, NetworkX, Scikit-learn

## Repository Structure

The project is organized using a standard Python package structure for clarity and scalability.