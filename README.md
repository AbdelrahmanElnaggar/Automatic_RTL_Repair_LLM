# Automatic_RTL_Repair_LLM
Investigating Automatic Bug Repair Using Large Language Models for Digital Hardware Design

# RTL_BugFixer_LLM Introduction
the full framework for automatic repair of hardware bugs using LLM

Register-transfer level ( RTL) bugs present critical challenges, impacting the functional correctness, security, and performance of system-on-chip ( SoC) designs. Detecting and repairing RTL bugs is traditionally a time-consuming process requiring skilled engineers, which significantly prolongs SoC development cycles and reduces vendor competitiveness. Given this complexity, there is a strong need for automated repair solutions capable of efficiently addressing RTL bugs to accelerate development timelines. In this thesis, we propose an automated framework leveraging a large language model (LLM) to repair RTL functional bugs. We explore various prompting techniques, including zero-shot, few-shot, and feedback approaches. Zero-shot relies solely on the LLM ’s pretrained knowledge, few-shot provides specific examples of RTL bug repairs, and feedback iteratively refines the LLM’s responses using outputs from prior iterations. Additionally, we investigate six prompting strategies, each incorporating varying levels of context to guide the LLM in the repair process. Our proposed framework operates on benchmarks without requiring prior knowledge of the bug’s type, location, or specific repair steps, better reflecting real-world scenarios than previous approaches. Results demonstrate the potential of LLM -driven automation, with the feedback approach achieving the highest repair success rate by fixing 26 out of 32 benchmarks (81.25%), while the best zero-shot and few-shot strategies repaired 23 out of 32 benchmarks (71.88%). These findings highlight the ability of current LLMs to consistently address RTL functional bugs, offering significant promise for streamlining SoC development by reducing the time and effort required for RTL bug detection and repair.

# How to Start Working with the Repository

## Prerequisites

Before you begin, ensure you have the following installed:

- Python version  3.9.19
- PyVerilog version 1.2.1
- Synopsys VCS (compiler version U-2023.03-SP1 Full64)
- OpenAI API (preferably gpt-4o-2024-05-13)

## Download the Cirfix Repository

1. Download the Cirfix repository from GitHub [here](https://github.com/hammad-a/verilog_repair).
2. Follow the steps provided on the repository page to download it.

## Add Cirfix Folders

1. After downloading the Cirfix repository, add the Cirfix folders to the `cirfix_benchmarks_code` directory in this repository.

## Running the Code

To run the code, you need to provide several command-line arguments to choose the required strategy. Below are the arguments:

### Command-Line Arguments

- `pandas_csv_path` (str): Path to the CSV file.
- `number_iterations` (int): Number of iterations to repeat passing the same prompt to GPT.
- `choose_file` (str): Choose file name to test. Use a specific file name or "all" to process all files.
- `scenario_ID` (int): Chooses the prompt scenario.
- `experiment_number` (str): Write the experiment number.
- `feedback_logic` (int): Choose your feedback logic.

### Example Command

```bash
python3 openapi_code.py output.csv <number_of_iterations> all <choosing_prompt_scenario> <experiment_number> <feedback_logic>




