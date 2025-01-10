import pandas as pd
import numpy as np
import sys, inspect, subprocess
import os
from optparse import OptionParser
import copy
import random
import time 
from datetime import datetime
import math
from openai import OpenAI
from secret_key import OPENAI_API_KEY
import MutationOp_class
from MutationOp_class import MutationOp
from MutationOp_class import get_output_mismatch, calc_candidate_fitness,extended_fl_for_study
import argparse
import json
import tiktoken
import shutil
from itertools import islice
import re
import pandas as pd
#import betterprompt

#Adding this to import libraries from another folder
sys.path.insert(1,'/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/cirfix_benchmarks_code/prototype')

from pyverilog.vparser.parser import parse, NodeNumbering
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.plyparser import ParseError
from pyverilog.vparser.ast import Node
import pyverilog.vparser.ast as vast
import fitness


#print(num_tokens_from_string("Hello world, let's test tiktoken.", "gpt-3.5-turbo"))

#to delete before sharing the code 
output_file_path = f"output_test.v"
output_directory = '/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/'
output_prompt_path=output_directory+"prompt"

if not os.path.exists(output_prompt_path):
    os.makedirs(output_prompt_path)
    print(f"Directory '{output_prompt_path}' created.")
else:
    print(f"Directory '{output_prompt_path}' already exists.")

jason_file_path = 'output_data.json'
#pandas_save_result_path = '/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/save_result2.csv'
#pandas_save_result_path = '/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/save_result_4o.csv'
#pandas_save_result_path = '/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/few_shot_save_result_4o.csv'
pandas_save_result_path = '/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/save_result2.csv'
#pandas_save_result_path = '/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/zeroshot_description_result_4o.csv'
pandas_save_feedback_result_path = '/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/save_feedback_result.csv'
baseline_save_results = '/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/save_baseline_result.csv'

# used in feedback
context = [ {'role':'system', 'content':"You are a helpful assistant for fixing Verilog and system Verilog code."} ]  # accumulate messages


few_shot_examples= f""" Follow the guidelines and examples provided below to learn how to spot and correct common issues. Use these as inspiration to identify bugs and apply the necessary fixes. 

    For detailed instructions and additional examples on bug repair, refer to the following guidelines: 
    

    1. **Conditional Statements**: 

        If there's a defect in conditional statements, invert the condition of the code block or correct the logical structure. 

        **Examples:** 

        - Example 1: 

            ```verilog 

            // Buggy line in conditional statement: Incorrect condition causes failure in signal check. 

            if (signal_ready == 1'b1) begin //buggy line 

            // Fixed line: Inverted the condition to correct signal check. 

            if (signal_ready == 1'b0) begin //fixed line 

            ``` 

        - Example 2: 

            ```verilog 

            // Buggy line in conditional statement: Missing else-if leads to incorrect control flow. 

            if (valid_data) begin //buggy line 

            // Fixed line: Added else-if for proper control flow. 

            else if (valid_data) begin //fixed line 

            ``` 

        - Example 3: 

            ```verilog 

            // Buggy line in conditional statement: Reset condition incorrect, leads to improper state assignment. 

            if (reset == 1'b1) state <= INIT; //buggy line 

            // Fixed line: Negated reset condition to fix state assignment. 

            if (reset == 1'b0) state <= INIT; //fixed line 

            ``` 

    

    2. **Sensitivity List**: 

        If the issue lies in the sensitivity list, verify that the always block triggers appropriately. This includes triggering the always block on signal's falling edge, rising edge, or any change to variables. 

    

        **Examples:** 

        - Example 1: 

            ```verilog 

            // Buggy line in sensitivity list: Incorrect edge trigger causes missed timing events. 

            always @(posedge clk) begin //buggy line 

            // Fixed line: Changed posedge to negedge for correct event trigger. 

            always @(negedge clk) begin //fixed line 

            ``` 

        - Example 2: 

            ```verilog 

            // Buggy line in sensitivity list: Edge condition for clk and reset incorrect, causing improper reset behavior. 

            always @(negedge clk or reset) begin //buggy line 

            // Fixed line: Corrected edge conditions for clk and reset. 

            always @(posedge clk or negedge reset) begin //fixed line 

            ``` 

        - Example 3: 

            ```verilog 

            // Buggy line in sensitivity list: Missing clock trigger for proper synchronization. 

            always @(data or reset) begin //buggy line 

            // Fixed line: Added proper clock trigger for synchronization. 

            always @(posedge clk or posedge reset) begin //fixed line 

            ``` 

    

    3. **Assignment Block**: 

        When dealing with assignment block defects, convert between blocking and non-blocking assignments as needed. 

    

        **Examples:** 

        - Example 1: 

            ```verilog 

            // Buggy line in assignment block: Incorrect blocking assignment causes race conditions. 

            out_signal = 1'b1; //buggy line 

            // Fixed line: Converted blocking to non-blocking to avoid race conditions. 

            out_signal <= 1'b1; //fixed line 

            ``` 

        - Example 2: 

            ```verilog 

            // Buggy line in assignment block: Non-blocking assignment causes delay in data registration. 

            data_reg <= input_data; //buggy line 

            // Fixed line: Changed to blocking assignment for immediate data update. 

            data_reg = input_data; //fixed line 

            ``` 

        - Example 3: 

            ```verilog 

            // Buggy line in assignment block: Blocking assignment causes incorrect timing for state update. 

            ready = ready_next; //buggy line 

            // Fixed line: Changed to non-blocking assignment for correct timing. 

            ready <= ready_next; //fixed line 

            ``` 

    

    4. **Numeric Value**: 

        For numeric value discrepancies, adjust the identifier by incrementing or decrementing it, or correct width mismatches. 

    

        **Examples:** 

        - Example 1: 

            ```verilog 

            // Buggy line in numeric value: Off-by-one error in parameter causes incorrect count. 

            parameter COUNT_MAX = 7; //buggy line 

            // Fixed line: Corrected the parameter value by incrementing. 

            parameter COUNT_MAX = 8; //fixed line 

            ``` 

        - Example 2: 

            ```verilog 

            // Buggy line in numeric value: Incorrect wire width leads to data truncation. 

            output wire [7:0] data_out; //buggy line 

            // Fixed line: Corrected wire width to prevent data truncation. 

            output wire [15:0] data_out; //fixed line 

            ``` 

        - Example 3: 

            ```verilog 

            // Buggy line in numeric value: Incorrect binary number leads to incorrect indexing. 

            assign index = 4'b111; //buggy line 

            // Fixed line: Corrected binary number. 

            assign index = 4'b0111; //fixed line 

            ``` 

    

    5. **Default Case in Case Statements**: 

        If the default case in a case statement is missing, add the default case to ensure proper operation. 

    

        **Examples:** 

        - Example 1: 

            ```verilog 

            // Buggy code in case statement: Missing default case leads to unexpected behavior. 

            case (current_state) 

            START: next_state = IDLE; 

            IDLE: next_state = EXEC; 

            endcase //buggy code 

            // Fixed code: Added default case for proper state handling. 

            case (current_state) 

            START: next_state = IDLE; 

            IDLE: next_state = EXEC; 

            default: next_state = ERROR; 

            endcase //fixed code 

            ``` 
    

    6. **Bitshifting Logic**: 

        If the bug is in bitshifting logic, verify the shift operator used and correct it. 

    

        **Examples:** 

        - Example 1: 

            ```verilog 

            // Buggy line in bitshifting logic: Incorrect operator used for bitshifting. 

            data_shifted = value > 2; //buggy line 

            // Fixed line: Corrected operator for bitshift operation. 

            data_shifted = value >> 2; //fixed line 

            ``` 

        - Example 2: 

            ```verilog 

            // Buggy line in bitshifting logic: Incorrect direction for bitshift operation. 

            result = input_data << 1; //buggy line 

            // Fixed line: Corrected direction of bitshift. 

            result = input_data >> 1; //fixed line 

            ``` 

    Use these guidelines and examples to fix the given buggy Verilog code 
    

    """ 










def directory_creation(scenario_ID,experiment_number):
    
    #directory_path=output_directory+"SC_ID_"+str(scenario_ID)+"/"+f_name
    directory_path=output_directory+"Experimental_output/"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"exp_num_"+str(experiment_number)+"/"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    directory_path=directory_path+"SC_ID_"+str(scenario_ID)+"/"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        

    return directory_path
        

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_string_gpto(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def basic_scenario(file_name,buggy_src_file_path,directory_path):

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_basic'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt1= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code}\n"
    prompt = f""" Your task is to fix the given buggy Verilog code and provide the complete, functioning version without adding any extra words, comments, or explanations in your response. 

    Your goal is to make one or more simple changes, focusing on the lines where the bug might originate, while ensuring that the overall logic and structure of the code remain unchanged.   

    The buggy Verilog code that needs to be fixed is as follows:  

    <<< \n{buggy_verilog_code}\n >>>.
    
    {few_shot_examples}
    
             """ 
        
    prompt1 = f"""  

        ### Your Task: ###  

        Fix the buggy Verilog code using the input information provided below to help you identify and resolve the bug.  Be sure to follow the instructions and the response format mentioned below to ensure accurate corrections. Use the guidelines and examples provided at the end of this prompt as hints to help you understand and correct the issues. 

        

        ### Instructions: ###  

        1. Follow the input information provided carefully to identify and fix the bug. 

        2.Follow the guidelines and examples provided at the end of this prompt to learn and understand how to spot and correct common issues. Use these examples as inspiration to identify the bugs and apply the necessary fixes. 

        3. Focus on making one or more simple changes in the lines where the bug might originate.  

        4. Ensure that the overall logic and structure of the code remain unchanged. 

        

        ### Response Format: ###  

        Provide the complete, functional Verilog code without any extra words, comments, or explanations. 

        

        ### Input Information: ###  

        1. The buggy Verilog code that needs to be fixed:  

        <<< {buggy_verilog_code} >>> 

        

        ### Repair Guidelines and Examples: ### 

        {few_shot_examples}

        """ 
    
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt1)) 
    #return prompt
    return prompt1


def bug_description_matching(file_name,buggy_src_file_path,directory_path,bug_description):

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_basic'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()


  
    prompt1 = f"""
            I have a piece of buggy Verilog code and I need help identifying the bug. I'll provide you with a list of potential bug descriptions; please choose the one that best matches the issue in the code. 

                Buggy Code:<<< {buggy_verilog_code} >>>

                
                List of Bug Descriptions: 

                1. Two separate numeric errors 
                2. Incorrect assignment 
                3. Counter incorrect sensitivity list 
                4. Else-if instead of if 
                5. Counter never reset 
                6. Negated if-condition 
                7. Branches of if-statement swapped 
                8. Default in case statement omitted 
                9. Assignment to next state omitted, default cases in case statement omitted 
                10. Assignment to next state omitted, state omitted from senslist 
                11. Blocking instead of nonblocking assignments 
                12. Blocking instead of nonblocking assignments 
                13. Negated if-condition 
                14. Incorrect sensitivity list 
                15. Three separate numeric errors 
                16. Hex instead of binary numbers 
                17. 1 bit instead of 4 bit output wire 
                18. Incorrect sensitivity list 
                19. Incorrect address assignment 
                20. Removed cmd_ack 
                21. Off-by-one error in loop 
                22. Incorrect assignment to out_ready 
                23. Not checking for buffer overflow during assignment 
                24. Logical instead of bitwise negation 
                25. Incorrect logic for bitshifting 
                26. `>` instead of `>>` for bitshifting 
                27. Incorrect instantiation of module 
                28. Insufficient register size for decimal values 
                29. Removed `@posedge reset` from senslist (sync vs async reset) 
                30. `wr_data_r` not reset correctly, `rd_data_r` assigned incorrectly 
                31. Numeric error in parameter 
                32. Default in case statement omitted 

                Please choose the bug description that best matches the issue in the provided code. 
            
            """


    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt1)) 
    #return prompt
    return prompt1




def testbench_changing(file_name,test_bench_path,directory_path):

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_basic'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(test_bench_path, 'r') as conf_file:
        testbench_verilog_code = conf_file.read()


  
    prompt_old = f"""
            modify this instrumented test bench to test for all possible input signal combinations and print input , output at everytime step Testbench:<<< {testbench_verilog_code} >>>
            Return the complete, corrected testbench code between the following pattern: start_code and end_code.

            """
    
    prompt1_old= f""" Modify the provided instrumented test bench to test all possible input signal combinations and print the input and output at every time step, without changing its existing structure. Testbench:<<< {testbench_verilog_code} >>>
    Return the complete, corrected testbench code between the following pattern: start_code and end_code."""

    prompt1= f"""Modify the provided instrumented test bench to print the input as well with the output at every time step, without changing its existing structure. Testbench:<<< {testbench_verilog_code} >>>  Return the complete, corrected testbench code between the following pattern: start_code and end_code. """ 

    prompt2 = f"""Modify the provided instrumented test bench to print both the input and the output at every time step, without altering its existing structure. Testbench: <<< {testbench_verilog_code} >>>. Return the complete, corrected testbench code between the following pattern: start_code and end_code.""" 

    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt2)) 
    #return prompt
    return prompt2


##################################################
#Added functions to get the diffference from oracle and simulation files
def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def find_differences_and_generate_prompt_output_only(original_file, buggy_file, max_differences=10):
    # Read the files
    #original_lines = read_file(original_file)
    #buggy_lines = read_file(buggy_file)
    original_lines = original_file
    buggy_lines = buggy_file

    # Get the heading (first line)
    heading = original_lines[0].strip()
    buggy_heading = buggy_lines[0].strip()

    # Ensure both files have the same heading
    if heading != buggy_heading:
        raise ValueError("The two files have different headings")

    # Split the headings into individual signal names
    heading_split = heading.split(',')

    # Initialize the prompt text
    prompt_text = "Identified differences between the original and buggy outputs:\n"

    # Counter for the number of differences found
    diff_count = 0

    # Compare each line after the heading
    for orig_line, buggy_line in zip(original_lines[1:], buggy_lines[1:]):
        orig_values = orig_line.strip().split(',')
        buggy_values_split = buggy_line.strip().split(',')

        # Only add details if there are differences
        if orig_values != buggy_values_split:
            prompt_text += f"At time {orig_values[0]}:\n"
            for i in range(1, len(orig_values)):  # Skip the time at index 0
                if orig_values[i] != buggy_values_split[i]:
                    prompt_text += f"  - Signal '{heading_split[i]}' should be {orig_values[i]}, but found {buggy_values_split[i]}.\n"
            prompt_text += "\n"
            diff_count += 1

        # Stop if the maximum number of differences is reached
        if diff_count >= max_differences:
            break

    return prompt_text


def find_differences_and_generate_prompt_new_output_only(original_file, buggy_file, max_differences=10, max_errors_per_line=10):
    # Read the files
    original_lines = original_file
    buggy_lines = buggy_file

    # Get the heading (first line)
    heading = original_lines[0].strip()
    buggy_heading = buggy_lines[0].strip()

    # Ensure both files have the same heading
    #if heading != buggy_heading:
        #raise ValueError("The two files have different headings")

    # Split the headings into individual signal names
    heading_split = heading.split(',')

    # Initialize the prompt text
    prompt_text = "Identified differences between the original and buggy outputs:\n"

    # Counter for the number of differences found
    diff_count = 0

    # Compare each line after the heading
    for orig_line, buggy_line in zip(original_lines[1:], buggy_lines[1:]):
        orig_values = orig_line.strip().split(',')
        buggy_values_split = buggy_line.strip().split(',')

        # Only add details if there are differences
        if orig_values != buggy_values_split:
            prompt_text += f"At time {orig_values[0]}:\n"
            
            error_count = 0  # Counter for errors in the current line
            for i in range(1, len(orig_values)):  # Skip the time at index 0
                if orig_values[i] != buggy_values_split[i]:
                    if error_count < max_errors_per_line:
                        prompt_text += f"  - Signal '{heading_split[i]}' should be {orig_values[i]}, but found {buggy_values_split[i]}.\n"
                        error_count += 1
                    else:
                        break
            prompt_text += "\n"
            diff_count += 1

        # Stop if the maximum number of differences is reached
        if diff_count >= max_differences:
            break

    return prompt_text


def find_differences_and_generate_prompt2_inout(original_file, buggy_file, max_differences=10):
    # Read the files
    original_lines = original_file
    buggy_lines = buggy_file

    # Get the heading (first line)
    heading = original_lines[0].strip()
    buggy_heading = buggy_lines[0].strip()

    #print("heading = ")
    #print(heading)
   

    # Ensure both files have the same heading
    #if heading != buggy_heading:
        #raise ValueError("The two files have different headings")

    # Split the headings into individual signal names
    heading_split = heading.split(',')

    # Initialize the prompt text
    prompt_text = "Identified differences between the original and buggy outputs:\n"

    # Counter for the number of differences found
    diff_count = 0

    # Compare each line after the heading
    for orig_line, buggy_line in zip(original_lines[1:], buggy_lines[1:]):
        orig_values = orig_line.strip().split(',')
        buggy_values_split = buggy_line.strip().split(',')

        # Only add details if there are differences
        if orig_values != buggy_values_split:
            prompt_text += f"At time {orig_values[0]} with inputs and outputs values "
            input_values = ", ".join([f"{heading_split[i]}={orig_values[i]}" for i in range(1, len(orig_values))])
            prompt_text += f"({input_values}):\n"
            
            for i in range(1, len(orig_values)):  # Skip the time at index 0
                if orig_values[i] != buggy_values_split[i]:
                    prompt_text += f"  - Signal '{heading_split[i]}' should be {orig_values[i]}, but found {buggy_values_split[i]}.\n"
            prompt_text += "\n"
            diff_count += 1

        # Stop if the maximum number of differences is reached
        if diff_count >= max_differences:
            break

    return prompt_text

def find_differences_and_generate_prompt3_inout(original_file, buggy_file, max_differences=10, max_errors_per_line=10):
    # Read the files
    original_lines = original_file
    buggy_lines = buggy_file

    # Get the heading (first line)
    heading = original_lines[0].strip()
    buggy_heading = buggy_lines[0].strip()

    # Split the headings into individual signal names
    heading_split = heading.split(',')

    # Initialize the prompt text
    prompt_text = "Identified differences between the original and buggy outputs:\n"

    # Counter for the number of differences found
    diff_count = 0

    # Compare each line after the heading
    for orig_line, buggy_line in zip(original_lines[1:], buggy_lines[1:]):
        orig_values = orig_line.strip().split(',')
        buggy_values_split = buggy_line.strip().split(',')

        # Only add details if there are differences
        if orig_values != buggy_values_split:
            prompt_text += f"At time {orig_values[0]} with inputs and outputs values "
            input_values = ", ".join([f"{heading_split[i]}={orig_values[i]}" for i in range(1, len(orig_values))])
            prompt_text += f"({input_values}):\n"

            error_count = 0  # Counter for errors in the current line
            for i in range(1, len(orig_values)):  # Skip the time at index 0
                if orig_values[i] != buggy_values_split[i]:
                    if error_count < max_errors_per_line:
                        prompt_text += f"  - Signal '{heading_split[i]}' should be {orig_values[i]}, but found {buggy_values_split[i]}.\n"
                        error_count += 1
                    else:
                        break

            prompt_text += "\n"
            diff_count += 1

        # Stop if the maximum number of differences is reached
        if diff_count >= max_differences:
            break

    return prompt_text



#Revised_code

def find_differences_and_generate_prompt4_inout(original_file, buggy_file, input_file, max_differences=10, max_errors_per_line=10):
    # Read the files
    original_lines = original_file
    buggy_lines = buggy_file
    input_lines = input_file

    # Get the headings (first line) from all files
    original_heading = original_lines[0].strip().split(',')
    buggy_heading = buggy_lines[0].strip().split(',')
    input_heading = input_lines[0].strip().split(',')

    # Initialize the prompt text
    prompt_text = "Identified differences between the original and buggy outputs:\n"

    # Create a dictionary to store inputs based on time steps
    input_dict = {}
    for input_line in input_lines[1:]:
        input_values = input_line.strip().split(',')
        time_step = input_values[0]
        input_dict[time_step] = input_values[1:]  # Store the input values excluding the time

    # Counter for the number of differences found
    diff_count = 0

    # Compare each line after the heading
    for orig_line, buggy_line in zip(original_lines[1:], buggy_lines[1:]):
        orig_values = orig_line.strip().split(',')
        buggy_values = buggy_line.strip().split(',')
        time_step = orig_values[0]

        # Only add details if there are differences
        if orig_values != buggy_values:
            # Get the input values for the current time step
            if time_step in input_dict:
                input_values = input_dict[time_step]
                input_text = ", ".join([f"{input_heading[i+1]}={input_values[i]}" for i in range(len(input_values))])
                # Notice that `i+1` is used to skip the time column in the input headings
            else:
                input_text = "Unknown inputs"

            prompt_text += f"At time {time_step} with inputs ({input_text}):\n"

            error_count = 0  # Counter for errors in the current line
            for i in range(1, len(orig_values)):  # Skip the time at index 0
                if orig_values[i] != buggy_values[i]:
                    if error_count < max_errors_per_line:
                        prompt_text += f"  - Output '{original_heading[i]}'  from the correct code should be {orig_values[i]},  but from the buggy code is {buggy_values[i]}.\n"
                        error_count += 1
                    else:
                        break

            prompt_text += "\n"
            diff_count += 1

        # Stop if the maximum number of differences is reached
        if diff_count >= max_differences:
            break

    return prompt_text


#optimized code (stream processing approach)
def find_differences_and_generate_prompt5_inout(original_file, buggy_file, input_file, max_differences=10, max_errors_per_line=10):
    # Open the files and process them line by line
    with open(original_file) as orig_f, open(buggy_file) as buggy_f, open(input_file) as input_f:
        # Read and split headings from the first line
        original_heading = orig_f.readline().strip().split(',')
        buggy_heading = buggy_f.readline().strip().split(',')
        input_heading = input_f.readline().strip().split(',')
        
        # Initialize the prompt text
        prompt_text = "Identified differences between the original and buggy outputs:\n"

        # Create a dictionary to store inputs based on time steps
        input_dict = {}
        for input_line in input_f:
            input_values = input_line.strip().split(',')
            time_step = input_values[0]
            input_dict[time_step] = input_values[1:]  # Store the input values excluding the time

        # Counter for the number of differences found
        diff_count = 0

        # Compare lines in original and buggy output files
        for orig_line, buggy_line in zip(orig_f, buggy_f):
            orig_values = orig_line.strip().split(',')
            buggy_values = buggy_line.strip().split(',')
            time_step = orig_values[0]

            # Only add details if there are differences
            if orig_values != buggy_values:
                # Get the input values for the current time step
                if time_step in input_dict:
                    input_values = input_dict[time_step]
                    input_text = ", ".join([f"{input_heading[i+1]}={input_values[i]}" for i in range(len(input_values))])
                else:
                    input_text = "Unknown inputs"

                prompt_text += f"At time {time_step} with inputs ({input_text}):\n"

                error_count = 0  # Counter for errors in the current line
                for i in range(1, len(orig_values)):  # Skip the time at index 0
                    if orig_values[i] != buggy_values[i]:
                        if error_count < max_errors_per_line:
                            prompt_text += f"  - Output '{original_heading[i]}'  from the correct code should be {orig_values[i]},  but from the buggy code is {buggy_values[i]}.\n"
                            error_count += 1
                        else:
                            break

                prompt_text += "\n"
                diff_count += 1

            # Stop if the maximum number of differences is reached
            if diff_count >= max_differences:
                break

    return prompt_text


def find_differences_and_generate_prompt6_inout(original_file, buggy_file, input_file, max_differences=10, max_errors_per_line=10):
    # Read the files
    original_lines = original_file
    buggy_lines = buggy_file
    input_lines = input_file
    
    '''
    print("original_lines")
    print(original_lines)
    print("buggy_lines")
    print(buggy_lines)
    print("input_lines")
    print(input_lines)
    '''
    


    # Get the heading (first line) from the original and input files
    output_heading = original_lines[0].strip()
    input_heading = input_lines[0].strip()
    print("input_heading")
    print(input_heading)
    
    # Split the headings into individual signal names
    output_heading_split = output_heading.split(',')
    input_heading_split = input_heading.split(',')

    # Initialize the prompt text
    prompt_text = "Identified differences between the original and buggy outputs:\n"

    # Counter for the number of differences found
    diff_count = 0

    # Create a dictionary to map time steps to their respective input values
    input_dict = {}
    for input_line in input_lines[1:]:
        input_values = input_line.strip().split(',')
        time_step = input_values[0]
        input_dict[time_step] = input_values
        #print("time_step")
        #print(time_step)
        #print("input_dict[time_step]=")
        #print(input_dict[time_step])

    

    
    # Compare each line after the heading
    for orig_line, buggy_line in zip(original_lines[1:], buggy_lines[1:]):
        orig_values = orig_line.strip().split(',')
        buggy_values_split = buggy_line.strip().split(',')
        time_step = orig_values[0]
        print("time_step=")
        print(time_step)
        

        # Only add details if there are differences
        if orig_values != buggy_values_split:
            prompt_text += f"At time step {time_step} with inputs: "

            # Fetch the corresponding input values for the current time step
            if time_step in input_dict:
                input_values = input_dict[time_step]
                input_value_str = ", ".join([f"{input_heading_split[i]}={input_values[i]}" for i in range(1, len(input_values))])
                prompt_text += f"({input_value_str}):\n"
                

            else:
                prompt_text += "(Unknown inputs):\n"

            
            error_count = 0  # Counter for errors in the current line
            for i in range(1, len(orig_values)):  # Skip the time at index 0
                if orig_values[i] != buggy_values_split[i]:
                    if error_count < max_errors_per_line:
                        prompt_text += f"  - The expected output for '{output_heading_split[i]}' from the correct code should be {orig_values[i]},  but the actual value is {buggy_values_split[i]} in the buggy code.\n"
                        error_count += 1
                    else:
                        break

            prompt_text += "\n"
            diff_count += 1

        # Stop if the maximum number of differences is reached
        if diff_count >= max_differences:
            break

    return prompt_text



 
#Gets all the line numbers for the code implicated by the FL.

'''    
def collect_lines_for_fl(self, ast):
    if ast.node_id in self.fault_loc_set:
        self.implicated_lines.add(ast.lineno)
        
    for c in ast.children():
        if c: self.collect_lines_for_fl(c)


def collect_lines_for_fl_modifed(self, ast, source_code_lines):
    """
    Traverses the AST to collect line numbers and corresponding lines of code 
    for nodes that are identified as fault-related.
    
    Parameters:
    - ast: The current node of the abstract syntax tree (AST) being processed.
    - source_code_lines: A list of the actual source code lines, where the index corresponds to the line number.
    """
    if ast.node_id in self.fault_loc_set:
        line_no = ast.lineno
        self.implicated_lines[line_no] = source_code_lines[line_no - 1]  # Store the actual line of code
        
    for c in ast.children():
        if c:
            self.collect_lines_for_fl(c, source_code_lines)

'''

def comp_tb_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

     # creating path for input file
    input_file_path = os.path.join(PROJ_DIR, "input_%s.txt" % TB_ID)

    with open(input_file_path, 'r') as conf_file:
        input_file = conf_file.readlines()

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code1 = codegen.visit(ast)
    
    # Read from the file
    with open(buggy_src_file_path, 'r') as f:
        src_code = f.readlines()
    
    print("src_code =")
    print(src_code)
    print("src_code len: %d" % len(src_code))
    
#################################################################################


    print("\n\n")
    
    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time ,oracle_lines,sim_lines= calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        #mutation_op.implicated_actual_lines = set()
        #mutation_op.collect_lines_for_fl(ast)
        #print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        #print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

        #mutation_op.implicated_actual_lines = dict()
        #source_code_lines = src_code.strip().split('\n')
        source_code_lines = src_code
        #print("source_code_lines =")
        #print(source_code_lines)
        mutation_op.collect_lines_for_fl_modifed(ast,source_code_lines)
       
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_actual_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_actual_lines))
        implicated_lines_summary = "\n".join([f"Line {line_no}: {source_code_lines[line_no - 1]}" for line_no in mutation_op.implicated_actual_lines])

        

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()



    # Example usage
    original_file = oracle_lines  # Replace with your file path
    buggy_file = sim_lines  # Replace with your file path
  
    

    # Generate the prompt with differences limited to 10 lines and 10 errors perline
    prompt_text = find_differences_and_generate_prompt6_inout(original_file, buggy_file,input_file, max_differences=10,max_errors_per_line=10)
    diff_path = 'differences_prompt.txt'

    diff_path_original = 'original_lines.txt'
    diff_path_sim = 'sim_lines.txt'
    implicated_lines_summary_data = "buggy_lines.txt"

    output_diff_path= os.path.join(directory_path, diff_path)

    output_diff_path1= os.path.join(directory_path, diff_path_original)
    output_diff_path2= os.path.join(directory_path, diff_path_sim)
    implicated_lines_summary_data_path= os.path.join(directory_path, implicated_lines_summary_data)

    # Write the prompt to a file or print it directly
    with open(output_diff_path, 'w') as f:
        #f.write(prompt_text)
        f.write(str(prompt_text))

    with open(output_diff_path1, 'w') as f:
      
        f.write(str(original_file))
    
    with open(output_diff_path2, 'w') as f:
        
        f.write(str(buggy_file))

    with open(implicated_lines_summary_data_path, 'w') as f:
        
        f.write(str(implicated_lines_summary))


    # Display the prompt
    print(prompt_text)

   # prompt = f""" Please provide the corrected Verilog code without adding any extra words, comments, or explanations in your response. Below is the buggy Verilog code that needs to be fixed:{buggy_verilog_code} After running this buggy Verilog code through the testbench, we obtained an output. We then compared this output with the oracle file, which represents the correct output expected from a bug-free version of the code. The comparison revealed the following differences between the output of the buggy code and the oracle: <{prompt_text}>""" 
    prompt_old = f"""Please provide the corrected Verilog code without adding any extra words, comments, or explanations in your response. Below is the buggy Verilog code that needs to be fixed:{buggy_verilog_code}
                After running this buggy Verilog code through the testbench, we obtained an output. We then compared this output with the oracle file, which represents the correct output expected from a bug-free version of the code. The comparison revealed the following differences between the output of the buggy code and the oracle: <{prompt_text}>
                Additionally, we have performed fault localization to identify lines in the code that may be contributing to the bug. Note that while the following lines are suspected as potential sources of the issue, the actual bug might also be in other lines of the code. Please consider all possible sources of the error while providing your corrected version of the Verilog code.
                The lines that are implicated are: 
                {implicated_lines_summary}"""
    
    prompt_old = f"""Please provide the corrected Verilog code without adding any extra words, comments, or explanations in your response. Below is the buggy Verilog code that needs to be fixed: {buggy_verilog_code}
        After running this buggy Verilog code through the testbench, we obtained an output. We then compared this output with the oracle file, which represents the correct output expected from a bug-free version of the code. The comparison revealed the following differences between the output of the buggy code and the oracle: <{prompt_text}>
        Additionally, we have performed fault localization to identify lines in the code that may be contributing to the bug. Note that while the following lines are suspected as potential sources of the issue, the actual bug might also be in other lines of the code. Please consider all possible sources of the error while providing your corrected version of the Verilog code.
        The fix should be simple, maintaining the same structure and logic of the code. The goal is to identify and implement a correction that will fix the bug and produce the same results as the oracle.
        The lines that are implicated are: 
        {implicated_lines_summary}"""

    ################################### New ################################
    prompt1= f""" Your task is to fix the given buggy Verilog code and provide the complete, functioning version without adding any extra words, comments, or explanations in your response. 
    Your goal is to make one or more simple changes, focusing on the lines where the bug might originate, while ensuring that the overall logic and structure of the code remain unchanged.  
    After simulating the buggy Verilog code with the testbench, the output signal at each time step was compared to the output signal values from the oracle file.  
    The oracle file represents the correct expected output from the bug-free version of the code.   
    The differences are formatted as follows:  
    For example:  
    At time step T with inputs: <input1> = <actual_input_value>, <input2> = <actual_input_value>, ...  

    The expected output for <output1> from the correct code should be <expected_output_value>, but the actual value is <actual_output_value> in the buggy code.  

    The comparison revealed the following differences:  

    <<< \n {prompt_text} \n >>> 


    The buggy Verilog code that needs to be fixed is as follows:  

    <<< \n{buggy_verilog_code}\n >>>. 

    {few_shot_examples}
    
    """ 


    prompt = f"""  

    ### Your Task: ###  

        Fix the buggy Verilog code using the input information provided below to help you identify and resolve the bug.  Be sure to follow the instructions and the response format mentioned below to ensure accurate corrections. Use the guidelines and examples provided at the end of this prompt as hints to help you understand and correct the issues. 
       

    ### Instructions: ###  

        1. Follow the input information provided carefully to identify and fix the bug. 
        2.Follow the guidelines and examples provided at the end of this prompt to learn and understand how to spot and correct common issues. Use these examples as inspiration to identify the bugs and apply the necessary fixes. 
        3. Focus on making one or more simple changes in the lines where the bug might originate.  
        4. Ensure that the overall logic and structure of the code remain unchanged. 
 
    ### Response Format: ###  

        Provide the complete, functional Verilog code without any extra words, comments, or explanations. 

    ### Input Information: ###  

        1. After simulating the buggy Verilog code with the testbench, the output signal at each time step was compared to the output signal values from the oracle file.  
        The oracle file represents the correct expected output from the bug-free version of the code.   
        The differences are formatted as follows:  

        For example:  

        At time step T with inputs: <input1> = <actual_input_value>, <input2> = <actual_input_value>, ...  

        The expected output for <output1> from the correct code should be <expected_output_value>, but the actual value is <actual_output_value> in the buggy code.  

        The comparison revealed the following differences:  

        <<< \n {prompt_text} \n >>> 



        2. The buggy Verilog code that needs to be fixed:  

        <<< {buggy_verilog_code} >>> 

    ### Repair Guidelines and Examples: ### 

        {few_shot_examples}

        """ 


    print(prompt)

    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt))

    return prompt






def detecting_lines_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

     # creating path for input file
    input_file_path = os.path.join(PROJ_DIR, "input_%s.txt" % TB_ID)

    with open(input_file_path, 'r') as conf_file:
        input_file = conf_file.readlines()

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code1 = codegen.visit(ast)
    
    # Read from the file
    with open(buggy_src_file_path, 'r') as f:
        src_code = f.readlines()
    
    print("src_code =")
    print(src_code)
    print("src_code len: %d" % len(src_code))
    
  
    
 
    
    
    '''
##################################################################################################
# start of the code to overcome the error 
    # try to overcome the changes between ast and src_code as ast has comments and src_code has no comments so the lines are different 
    #why not make src code read the same file 
    codegen2= ASTCodeGenerator()
    temp_f_name=file_name.split('.v')[0]
    directory_path_temp=directory_path+temp_f_name
    temp_src_path = 'temp_src_code.v'
    temp_temp_path= os.path.join(directory_path_temp, temp_src_path)
    if not os.path.exists(directory_path_temp):
        os.makedirs(directory_path_temp)
        print(f"Directory '{directory_path_temp}' created.")
    else:
        print(f"Directory '{directory_path_temp}' already exists.")

    with open(temp_temp_path, 'w') as f:
      
        f.write(str(src_code1))

    ast, directives = parse([temp_temp_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    ast.show()
    print(ast)
    src_code = codegen2.visit(ast)
    print("source_code2 = ")
    print(src_code)
    #end of the code 
    '''
#################################################################################


    print("\n\n")
    
    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time ,oracle_lines,sim_lines= calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        #mutation_op.implicated_actual_lines = set()
        #mutation_op.collect_lines_for_fl(ast)
        #print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        #print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

        #mutation_op.implicated_actual_lines = dict()
        #source_code_lines = src_code.strip().split('\n')
        source_code_lines = src_code
        #print("source_code_lines =")
        #print(source_code_lines)
        mutation_op.collect_lines_for_fl_modifed(ast,source_code_lines)
       
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_actual_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_actual_lines))
        implicated_lines_summary = "\n".join([f"Line {line_no}: {source_code_lines[line_no - 1]}" for line_no in mutation_op.implicated_actual_lines])

        

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()



    # Example usage
    original_file = oracle_lines  # Replace with your file path
    buggy_file = sim_lines  # Replace with your file path
  
    

    # Generate the prompt with differences limited to 10 lines and 10 errors perline
    prompt_text = find_differences_and_generate_prompt6_inout(original_file, buggy_file,input_file, max_differences=10,max_errors_per_line=10)
    diff_path = 'differences_prompt.txt'

    diff_path_original = 'original_lines.txt'
    diff_path_sim = 'sim_lines.txt'
    implicated_lines_summary_data = "buggy_lines.txt"

    output_diff_path= os.path.join(directory_path, diff_path)

    output_diff_path1= os.path.join(directory_path, diff_path_original)
    output_diff_path2= os.path.join(directory_path, diff_path_sim)
    implicated_lines_summary_data_path= os.path.join(directory_path, implicated_lines_summary_data)

    # Write the prompt to a file or print it directly
    with open(output_diff_path, 'w') as f:
        #f.write(prompt_text)
        f.write(str(prompt_text))

    with open(output_diff_path1, 'w') as f:
      
        f.write(str(original_file))
    
    with open(output_diff_path2, 'w') as f:
        
        f.write(str(buggy_file))

    with open(implicated_lines_summary_data_path, 'w') as f:
        
        f.write(str(implicated_lines_summary))


    # Display the prompt
    print(prompt_text)

   # prompt = f""" Please provide the corrected Verilog code without adding any extra words, comments, or explanations in your response. Below is the buggy Verilog code that needs to be fixed:{buggy_verilog_code} After running this buggy Verilog code through the testbench, we obtained an output. We then compared this output with the oracle file, which represents the correct output expected from a bug-free version of the code. The comparison revealed the following differences between the output of the buggy code and the oracle: <{prompt_text}>""" 
    prompt_old = f"""Please provide the corrected Verilog code without adding any extra words, comments, or explanations in your response. Below is the buggy Verilog code that needs to be fixed:{buggy_verilog_code}
                After running this buggy Verilog code through the testbench, we obtained an output. We then compared this output with the oracle file, which represents the correct output expected from a bug-free version of the code. The comparison revealed the following differences between the output of the buggy code and the oracle: <{prompt_text}>
                Additionally, we have performed fault localization to identify lines in the code that may be contributing to the bug. Note that while the following lines are suspected as potential sources of the issue, the actual bug might also be in other lines of the code. Please consider all possible sources of the error while providing your corrected version of the Verilog code.
                The lines that are implicated are: 
                {implicated_lines_summary}"""
    
    prompt_old = f"""Please provide the corrected Verilog code without adding any extra words, comments, or explanations in your response. Below is the buggy Verilog code that needs to be fixed: {buggy_verilog_code}
        After running this buggy Verilog code through the testbench, we obtained an output. We then compared this output with the oracle file, which represents the correct output expected from a bug-free version of the code. The comparison revealed the following differences between the output of the buggy code and the oracle: <{prompt_text}>
        Additionally, we have performed fault localization to identify lines in the code that may be contributing to the bug. Note that while the following lines are suspected as potential sources of the issue, the actual bug might also be in other lines of the code. Please consider all possible sources of the error while providing your corrected version of the Verilog code.
        The fix should be simple, maintaining the same structure and logic of the code. The goal is to identify and implement a correction that will fix the bug and produce the same results as the oracle.
        The lines that are implicated are: 
        {implicated_lines_summary}"""

    ################################### New ################################
    prompt1 = f""" Your task is to fix the given buggy Verilog code and provide the complete, functioning version without adding any extra words, comments, or explanations in your response. 
    Your goal is to make one or more simple changes, focusing on the lines where the bug might originate, while ensuring that the overall logic and structure of the code remain unchanged.  
    Fault localization has been performed to identify lines that may be contributing to the bug. While the following lines are suspected as potential sources, the bug may exist in other parts of the code as well. Consider all possible sources of the error when providing your corrected version of the Verilog code.   
    Lines identified by fault localization that might contain the bug are:   

    <<< \n {implicated_lines_summary} \n >>>.  

    The buggy Verilog code that needs to be fixed is as follows:  

    <<< \n{buggy_verilog_code}\n >>>. 

    
    {few_shot_examples}

    """ 

    prompt = f"""  

    ### Your Task: ###  

        Fix the buggy Verilog code using the input information provided below to help you identify and resolve the bug.  Be sure to follow the instructions and the response format mentioned below to ensure accurate corrections. Use the guidelines and examples provided at the end of this prompt as hints to help you understand and correct the issues. 
       

    ### Instructions: ###  

        1. Follow the input information provided carefully to identify and fix the bug. 
        2.Follow the guidelines and examples provided at the end of this prompt to learn and understand how to spot and correct common issues. Use these examples as inspiration to identify the bugs and apply the necessary fixes. 
        3. Focus on making one or more simple changes in the lines where the bug might originate.  
        4. Ensure that the overall logic and structure of the code remain unchanged. 
 
    ### Response Format: ###  

        Provide the complete, functional Verilog code without any extra words, comments, or explanations. 


    ### Input Information: ###  

        1. Fault localization has been performed to identify lines that may be contributing to the bug. While the following lines are suspected as potential sources, the bug may exist in other parts of the code as well. Consider all possible sources of the error when providing your corrected version of the Verilog code.   

        Lines identified by fault localization that might contain the bug are:   

        <<< \n {implicated_lines_summary} \n >>> 
    

        2. The buggy Verilog code that needs to be fixed:  

            <<< {buggy_verilog_code} >>> 

    ### Repair Guidelines and Examples: ### 

        {few_shot_examples}

    """ 

    print(prompt)

    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt))

    return prompt



def lines_with_Fault(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

     # creating path for input file
    input_file_path = os.path.join(PROJ_DIR, "input_%s.txt" % TB_ID)

    with open(input_file_path, 'r') as conf_file:
        input_file = conf_file.readlines()

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code1 = codegen.visit(ast)
    
    # Read from the file
    with open(buggy_src_file_path, 'r') as f:
        src_code = f.readlines()
    
    print("src_code =")
    print(src_code)
    print("src_code len: %d" % len(src_code))
    
#################################################################################


    print("\n\n")
    
    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time ,oracle_lines,sim_lines= calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        #mutation_op.implicated_actual_lines = set()
        #mutation_op.collect_lines_for_fl(ast)
        #print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        #print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

        #mutation_op.implicated_actual_lines = dict()
        #source_code_lines = src_code.strip().split('\n')
        source_code_lines = src_code
        #print("source_code_lines =")
        #print(source_code_lines)
        mutation_op.collect_lines_for_fl_modifed(ast,source_code_lines)
       
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_actual_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_actual_lines))
        implicated_lines_summary = "\n".join([f"Line {line_no}: {source_code_lines[line_no - 1]}" for line_no in mutation_op.implicated_actual_lines])

        

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()



    # Example usage
    original_file = oracle_lines  # Replace with your file path
    buggy_file = sim_lines  # Replace with your file path
  
    

    # Generate the prompt with differences limited to 10 lines and 10 errors perline
    prompt_text = find_differences_and_generate_prompt6_inout(original_file, buggy_file,input_file, max_differences=10,max_errors_per_line=10)
    diff_path = 'differences_prompt.txt'

    diff_path_original = 'original_lines.txt'
    diff_path_sim = 'sim_lines.txt'
    implicated_lines_summary_data = "buggy_lines.txt"

    output_diff_path= os.path.join(directory_path, diff_path)

    output_diff_path1= os.path.join(directory_path, diff_path_original)
    output_diff_path2= os.path.join(directory_path, diff_path_sim)
    implicated_lines_summary_data_path= os.path.join(directory_path, implicated_lines_summary_data)

    # Write the prompt to a file or print it directly
    with open(output_diff_path, 'w') as f:
        #f.write(prompt_text)
        f.write(str(prompt_text))

    with open(output_diff_path1, 'w') as f:
      
        f.write(str(original_file))
    
    with open(output_diff_path2, 'w') as f:
        
        f.write(str(buggy_file))

    with open(implicated_lines_summary_data_path, 'w') as f:
        
        f.write(str(implicated_lines_summary))


    # Display the prompt
    print(prompt_text)

   # prompt = f""" Please provide the corrected Verilog code without adding any extra words, comments, or explanations in your response. Below is the buggy Verilog code that needs to be fixed:{buggy_verilog_code} After running this buggy Verilog code through the testbench, we obtained an output. We then compared this output with the oracle file, which represents the correct output expected from a bug-free version of the code. The comparison revealed the following differences between the output of the buggy code and the oracle: <{prompt_text}>""" 
    prompt_old = f"""Please provide the corrected Verilog code without adding any extra words, comments, or explanations in your response. Below is the buggy Verilog code that needs to be fixed:{buggy_verilog_code}
                After running this buggy Verilog code through the testbench, we obtained an output. We then compared this output with the oracle file, which represents the correct output expected from a bug-free version of the code. The comparison revealed the following differences between the output of the buggy code and the oracle: <{prompt_text}>
                Additionally, we have performed fault localization to identify lines in the code that may be contributing to the bug. Note that while the following lines are suspected as potential sources of the issue, the actual bug might also be in other lines of the code. Please consider all possible sources of the error while providing your corrected version of the Verilog code.
                The lines that are implicated are: 
                {implicated_lines_summary}"""
    
    prompt_old = f"""Please provide the corrected Verilog code without adding any extra words, comments, or explanations in your response. Below is the buggy Verilog code that needs to be fixed: {buggy_verilog_code}
        After running this buggy Verilog code through the testbench, we obtained an output. We then compared this output with the oracle file, which represents the correct output expected from a bug-free version of the code. The comparison revealed the following differences between the output of the buggy code and the oracle: <{prompt_text}>
        Additionally, we have performed fault localization to identify lines in the code that may be contributing to the bug. Note that while the following lines are suspected as potential sources of the issue, the actual bug might also be in other lines of the code. Please consider all possible sources of the error while providing your corrected version of the Verilog code.
        The fix should be simple, maintaining the same structure and logic of the code. The goal is to identify and implement a correction that will fix the bug and produce the same results as the oracle.
        The lines that are implicated are: 
        {implicated_lines_summary}"""

    ################################### New ################################
    prompt1 = f""" Your task is to fix the given buggy Verilog code and provide the complete, functioning version without adding any extra words, comments, or explanations in your response. 
    Your goal is to make one or more simple changes, focusing on the lines where the bug might originate, while ensuring that the overall logic and structure of the code remain unchanged.  
    After simulating the buggy Verilog code with the testbench, the output signal at each time step was compared to the output signal values from the oracle file.  
    The oracle file represents the correct expected output from the bug-free version of the code.   
    The differences are formatted as follows:  
    For example:  
    At time step T with inputs: <input1> = <actual_input_value>, <input2> = <actual_input_value>, ...  

    The expected output for <output1> from the correct code should be <expected_output_value>, but the actual value is <actual_output_value> in the buggy code.  

    The comparison revealed the following differences:  

    <<< \n {prompt_text} \n >>> 

    Fault localization has been performed to identify lines that may be contributing to the bug. While the following lines are suspected as potential sources, the bug may exist in other parts of the code as well. Consider all possible sources of the error when providing your corrected version of the Verilog code.   
    Lines identified by fault localization that might contain the bug are:   

    <<< \n {implicated_lines_summary} \n >>>.  

    The buggy Verilog code that needs to be fixed is as follows:  

    <<< \n{buggy_verilog_code}\n >>>. 

    {few_shot_examples}

    """ 


    prompt = f"""  

    ### Your Task: ###  

        Fix the buggy Verilog code using the input information provided below to help you identify and resolve the bug.  Be sure to follow the instructions and the response format mentioned below to ensure accurate corrections. Use the guidelines and examples provided at the end of this prompt as hints to help you understand and correct the issues. 
       

    ### Instructions: ###  

        1. Follow the input information provided carefully to identify and fix the bug. 
        2.Follow the guidelines and examples provided at the end of this prompt to learn and understand how to spot and correct common issues. Use these examples as inspiration to identify the bugs and apply the necessary fixes. 
        3. Focus on making one or more simple changes in the lines where the bug might originate.  
        4. Ensure that the overall logic and structure of the code remain unchanged. 
 
    ### Response Format: ###  

        Provide the complete, functional Verilog code without any extra words, comments, or explanations. 

    

    ### Input Information: ###  

        1. After simulating the buggy Verilog code with the testbench, the output signal at each time step was compared to the output signal values from the oracle file.  
        The oracle file represents the correct expected output from the bug-free version of the code.   
        The differences are formatted as follows:  

        For example:  

        At time step T with inputs: <input1> = <actual_input_value>, <input2> = <actual_input_value>, ...  

        The expected output for <output1> from the correct code should be <expected_output_value>, but the actual value is <actual_output_value> in the buggy code.  

        The comparison revealed the following differences:  

        <<< \n {prompt_text} \n >>> 

        

        2. Fault localization has been performed to identify lines that may be contributing to the bug. While the following lines are suspected as potential sources, the bug may exist in other parts of the code as well. Consider all possible sources of the error when providing your corrected version of the Verilog code.   
        Lines identified by fault localization that might contain the bug are:   

        <<< \n {implicated_lines_summary} \n >>> 

        

        3. The buggy Verilog code that needs to be fixed:  

        <<< {buggy_verilog_code} >>> 

    
    ### Repair Guidelines and Examples: ### 

        {few_shot_examples}

    """ 


    print(prompt)

    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt))

    return prompt



def all_input_information(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

     # creating path for input file
    input_file_path = os.path.join(PROJ_DIR, "input_%s.txt" % TB_ID)

    with open(input_file_path, 'r') as conf_file:
        input_file = conf_file.readlines()

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code1 = codegen.visit(ast)
    
    # Read from the file
    with open(buggy_src_file_path, 'r') as f:
        src_code = f.readlines()
    
    print("src_code =")
    print(src_code)
    print("src_code len: %d" % len(src_code))
    
#################################################################################


    print("\n\n")
    
    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time ,oracle_lines,sim_lines= calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        #mutation_op.implicated_actual_lines = set()
        #mutation_op.collect_lines_for_fl(ast)
        #print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        #print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

        #mutation_op.implicated_actual_lines = dict()
        #source_code_lines = src_code.strip().split('\n')
        source_code_lines = src_code
        #print("source_code_lines =")
        #print(source_code_lines)
        mutation_op.collect_lines_for_fl_modifed(ast,source_code_lines)
       
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_actual_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_actual_lines))
        implicated_lines_summary = "\n".join([f"Line {line_no}: {source_code_lines[line_no - 1]}" for line_no in mutation_op.implicated_actual_lines])

        

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()



    # Example usage
    original_file = oracle_lines  # Replace with your file path
    buggy_file = sim_lines  # Replace with your file path
  
    

    # Generate the prompt with differences limited to 10 lines and 10 errors perline
    prompt_text = find_differences_and_generate_prompt6_inout(original_file, buggy_file,input_file, max_differences=10,max_errors_per_line=10)
    diff_path = 'differences_prompt.txt'

    diff_path_original = 'original_lines.txt'
    diff_path_sim = 'sim_lines.txt'
    implicated_lines_summary_data = "buggy_lines.txt"

    output_diff_path= os.path.join(directory_path, diff_path)

    output_diff_path1= os.path.join(directory_path, diff_path_original)
    output_diff_path2= os.path.join(directory_path, diff_path_sim)
    implicated_lines_summary_data_path= os.path.join(directory_path, implicated_lines_summary_data)

    # Write the prompt to a file or print it directly
    with open(output_diff_path, 'w') as f:
        #f.write(prompt_text)
        f.write(str(prompt_text))

    with open(output_diff_path1, 'w') as f:
      
        f.write(str(original_file))
    
    with open(output_diff_path2, 'w') as f:
        
        f.write(str(buggy_file))

    with open(implicated_lines_summary_data_path, 'w') as f:
        
        f.write(str(implicated_lines_summary))


    # Display the prompt
    print(prompt_text)

    prompt1 = f""" Your task is to fix the given buggy Verilog code and provide the complete, functioning version without adding any extra words, comments, or explanations in your response. 
    Your goal is to make one or more simple changes, focusing on the lines where the bug might originate, while ensuring that the overall logic and structure of the code remain unchanged.  
    After comparing the output of the correct and buggy code, a mismatch was identified between the two outputs. The elements in the code responsible for this mismatch have been identified as potential sources of the bug. These elements, which may include input ports, output ports, registers, wires, or data types, are: 
    <<<{tmp_mismatch_set}>>>. 

    After simulating the buggy Verilog code with the testbench, the output signal at each time step was compared to the output signal values from the oracle file.  
    The oracle file represents the correct expected output from the bug-free version of the code.   
    The differences are formatted as follows:  
    For example:  

    At time step T with inputs: <input1> = <actual_input_value>, <input2> = <actual_input_value>, ...  

    The expected output for <output1> from the correct code should be <expected_output_value>, but the actual value is <actual_output_value> in the buggy code.  

    The comparison revealed the following differences:  

    <<< \n {prompt_text} \n >>> 


    Fault localization has been performed to identify lines that may be contributing to the bug. While the following lines are suspected as potential sources, the bug may exist in other parts of the code as well. Consider all possible sources of the error when providing your corrected version of the Verilog code.   
    Lines identified by fault localization that might contain the bug are:   

    <<< \n {implicated_lines_summary} \n >>>.  

    The buggy Verilog code that needs to be fixed is as follows:  

    <<< \n{buggy_verilog_code}\n >>>. 
    
    {few_shot_examples}
    
    """ 


    prompt= f"""  

    ### Your Task: ###  

        Fix the buggy Verilog code using the input information provided below to help you identify and resolve the bug.  Be sure to follow the instructions and the response format mentioned below to ensure accurate corrections. Use the guidelines and examples provided at the end of this prompt as hints to help you understand and correct the issues. 
       

    ### Instructions: ###  

        1. Follow the input information provided carefully to identify and fix the bug. 
        2. Follow the guidelines and examples provided at the end of this prompt to learn and understand how to spot and correct common issues. Use these examples as inspiration to identify the bugs and apply the necessary fixes. 
        3. Focus on making one or more simple changes in the lines where the bug might originate.  
        4. Ensure that the overall logic and structure of the code remain unchanged. 
 
    ### Response Format: ###  

        Provide the complete, functional Verilog code without any extra words, comments, or explanations. 

    

    ### Input Information: ###  

        1. After comparing the output of the correct and buggy code, a mismatch was identified between the two outputs. The elements in the code responsible for this mismatch have been identified as potential sources of the bug. These elements, which may include input ports, output ports, registers, wires, or data types, are: <<<{tmp_mismatch_set}>>> 

        

        2. After simulating the buggy Verilog code with the testbench, the output signal at each time step was compared to the output signal values from the oracle file.  
        The oracle file represents the correct expected output from the bug-free version of the code.   
        The differences are formatted as follows:  
        For example:  

        At time step T with inputs: <input1> = <actual_input_value>, <input2> = <actual_input_value>, ...  

        The expected output for <output1> from the correct code should be <expected_output_value>, but the actual value is <actual_output_value> in the buggy code.  

        The comparison revealed the following differences:  

        <<< \n {prompt_text} \n >>> 

        

        3. Fault localization has been performed to identify lines that may be contributing to the bug. While the following lines are suspected as potential sources, the bug may exist in other parts of the code as well. Consider all possible sources of the error when providing your corrected version of the Verilog code.   
        Lines identified by fault localization that might contain the bug are:   

        <<< \n {implicated_lines_summary} \n >>> 

        

        4. The buggy Verilog code that needs to be fixed:  

        <<< {buggy_verilog_code} >>> 


    ### Repair Guidelines and Examples: ### 

        {few_shot_examples}

    """ 


    print(prompt)

    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt))

    return prompt


##################################

def all_input_feedback(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,current_trial):

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

     # creating path for input file
    input_file_path = os.path.join(PROJ_DIR, "input_%s.txt" % TB_ID)

    with open(input_file_path, 'r') as conf_file:
        input_file = conf_file.readlines()

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code1 = codegen.visit(ast)
    
    # Read from the file
    with open(buggy_src_file_path, 'r') as f:
        src_code = f.readlines()
    
    print("src_code =")
    print(src_code)
    print("src_code len: %d" % len(src_code))
    
#################################################################################


    print("\n\n")
    
    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time ,oracle_lines,sim_lines= calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        #mutation_op.implicated_actual_lines = set()
        #mutation_op.collect_lines_for_fl(ast)
        #print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        #print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

        #mutation_op.implicated_actual_lines = dict()
        #source_code_lines = src_code.strip().split('\n')
        source_code_lines = src_code
        #print("source_code_lines =")
        #print(source_code_lines)
        mutation_op.collect_lines_for_fl_modifed(ast,source_code_lines)
       
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_actual_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_actual_lines))
        implicated_lines_summary = "\n".join([f"Line {line_no}: {source_code_lines[line_no - 1]}" for line_no in mutation_op.implicated_actual_lines])

        

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()



    # Example usage
    original_file = oracle_lines  # Replace with your file path
    buggy_file = sim_lines  # Replace with your file path
  
    

    # Generate the prompt with differences limited to 10 lines and 10 errors perline
    prompt_text = find_differences_and_generate_prompt6_inout(original_file, buggy_file,input_file, max_differences=10,max_errors_per_line=10)
    diff_path = 'differences_prompt.txt'

    diff_path_original = 'original_lines.txt'
    diff_path_sim = 'sim_lines.txt'
    implicated_lines_summary_data = "buggy_lines.txt"

    output_diff_path= os.path.join(directory_path, diff_path)

    output_diff_path1= os.path.join(directory_path, diff_path_original)
    output_diff_path2= os.path.join(directory_path, diff_path_sim)
    implicated_lines_summary_data_path= os.path.join(directory_path, implicated_lines_summary_data)

    # Write the prompt to a file or print it directly
    with open(output_diff_path, 'w') as f:
        #f.write(prompt_text)
        f.write(str(prompt_text))

    with open(output_diff_path1, 'w') as f:
      
        f.write(str(original_file))
    
    with open(output_diff_path2, 'w') as f:
        
        f.write(str(buggy_file))

    with open(implicated_lines_summary_data_path, 'w') as f:
        
        f.write(str(implicated_lines_summary))


    # Display the prompt
    print(prompt_text)

    if(current_trial==1):
        prompt_final= f"""  
        Simulated output differences: After simulating the buggy Verilog code with the testbench, the output signal at each time step was compared to the output signal values from the oracle file.  
        The oracle file represents the correct expected output from the bug-free version of the code.   
        The differences are formatted as follows:  
        For example:  

        At time step T with inputs: <input1> = <actual_input_value>, <input2> = <actual_input_value>, ...  

        The expected output for <output1> from the correct code should be <expected_output_value>, but the actual value is <actual_output_value> in the buggy code.  

        The comparison revealed the following differences:  

        <<< \n {prompt_text} \n >>> """

    if(current_trial==2 or current_trial==3):
        prompt_final= f"""  
        1- After simulating the buggy Verilog code with the testbench, the output signal at each time step was compared to the output signal values from the oracle file.  
        The oracle file represents the correct expected output from the bug-free version of the code.   
        The differences are formatted as follows:  
        For example:  

        At time step T with inputs: <input1> = <actual_input_value>, <input2> = <actual_input_value>, ...  

        The expected output for <output1> from the correct code should be <expected_output_value>, but the actual value is <actual_output_value> in the buggy code.  

        The comparison revealed the following differences:  

        <<< \n {prompt_text} \n >>>

        2- Fault localization has been performed to identify lines that may be contributing to the bug. While the following lines are suspected as potential sources, the bug may exist in other parts of the code as well. Consider all possible sources of the error when providing your corrected version of the Verilog code.

        Lines identified by fault localization that might contain the bug are:
        <<< \n {implicated_lines_summary} \n >>>
        """


    
    prompt= f"""  

        ### Your Task: ###  

            Fix the buggy Verilog code using the input information provided below to help you identify and resolve the bug.  Be sure to follow the instructions and the response format mentioned below to ensure accurate corrections. Use the guidelines and examples provided at the end of this prompt as hints to help you understand and correct the issues. 
        

        ### Instructions: ###  

            1. Follow the input information provided carefully to identify and fix the bug. 
            2. Follow the guidelines and examples provided at the end of this prompt to learn and understand how to spot and correct common issues. Use these examples as inspiration to identify the bugs and apply the necessary fixes. 
            3. Focus on making one or more simple changes in the lines where the bug might originate.  
            4. Ensure that the overall logic and structure of the code remain unchanged. 
    
        ### Response Format: ###  

            Provide the complete, functional Verilog code without any extra words, comments, or explanations. 

        

        ### Input Information: ###  

            1. After comparing the output of the correct and buggy code, a mismatch was identified between the two outputs. The elements in the code responsible for this mismatch have been identified as potential sources of the bug. These elements, which may include input ports, output ports, registers, wires, or data types, are: <<<{tmp_mismatch_set}>>> 

            

            2. After simulating the buggy Verilog code with the testbench, the output signal at each time step was compared to the output signal values from the oracle file.  
            The oracle file represents the correct expected output from the bug-free version of the code.   
            The differences are formatted as follows:  
            For example:  

            At time step T with inputs: <input1> = <actual_input_value>, <input2> = <actual_input_value>, ...  

            The expected output for <output1> from the correct code should be <expected_output_value>, but the actual value is <actual_output_value> in the buggy code.  

            The comparison revealed the following differences:  

            <<< \n {prompt_text} \n >>> 

            

            3. Fault localization has been performed to identify lines that may be contributing to the bug. While the following lines are suspected as potential sources, the bug may exist in other parts of the code as well. Consider all possible sources of the error when providing your corrected version of the Verilog code.   
            Lines identified by fault localization that might contain the bug are:   

            <<< \n {implicated_lines_summary} \n >>> 

            

            4. The buggy Verilog code that needs to be fixed:  

            <<< {buggy_verilog_code} >>> 


            ### Repair Guidelines and Examples: ### 

                {few_shot_examples}

        """ 

    print("prompt_text = ")
    print(prompt_text)

    print("implicated_lines_summary = ")
    print(implicated_lines_summary)
    #with open(output_file_path, 'w') as file_a:
        #file_a.write(str(prompt))

    return prompt_final



##################################
    
def bug_description_scenario(file_name,buggy_src_file_path,directory_path,bug_description):

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_basic'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code} ,given that the bug description is {bug_description} \n"
    #prompt = f"Provide the complete functioning code without adding extra words, comments, or explanations. Fix the given buggy Verilog code:\n{buggy_verilog_code}, considering the bug description: {bug_description}\n"
    prompt1 = f"Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: {bug_description}, fix the following buggy Verilog code:\n{buggy_verilog_code}\n"
    
    prompt2=f"""Your task is to correct the specified bug in the given Verilog code while maintaining its functionality. The description of the bug is <<< {bug_description}>>>. It is important to provide the complete functioning code without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>> Your task is to identify and implement the necessary correction(s) to resolve the bug while adhering to the specified guidelines. """
    
    prompt3=f""" Your task is to provide the complete functioning code as a hardware engineer without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The description of the bug is <<< {bug_description}>>>. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>>  """
    
    

    prompt4_old = f"""Your task is to fix the given buggy Verilog code and provide the complete, functioning version without adding any extra words, comments, or explanations in your response.
    Your goal is to make one or more simple changes, focusing on the lines where the bug might originate, while ensuring that the overall logic and structure of the code remain unchanged.
    The description of the bug is as follows:  <<< {bug_description} >>>.

    The buggy Verilog code that needs to be fixed is as follows:

    <<< \n{buggy_verilog_code}\n >>>."""

    prompt4 = f"""Your task is to fix the provided buggy Verilog code and provide a complete, functioning version without adding any extra words, comments, or explanations in your response. Ensure that you understand both the code and the bug description before making the changes.

    Focus on the lines where the bug might originate, and make sure that the overall logic and structure of the code remain unchanged.

    The description of the bug is as follows: <<< {bug_description} >>>.

    The buggy Verilog code that needs to be fixed is as follows:

    <<< \n{buggy_verilog_code}\n >>>."""



    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt4)) 
    #return prompt
    return prompt4


def scenario_mismatch(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")

    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time,_,_= calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_mismatch'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    prompt= f""" Your task is to fix the given buggy Verilog code and provide the complete, functioning version without adding any extra words, comments, or explanations in your response.  
    Your goal is to make one or more simple changes, focusing on the lines where the bug might originate, while ensuring that the overall logic and structure of the code remain unchanged.  
    After comparing the output of the correct and buggy code, a mismatch was identified between the two outputs. The elements in the code responsible for this mismatch have been identified as potential sources of the bug.
    These elements, which may include input ports, output ports, registers, wires, or data types, are: <<<{tmp_mismatch_set}>>>.   
    The buggy Verilog code that needs to be fixed is as follows:  

    <<< \n{buggy_verilog_code}\n >>>. 
     
      {few_shot_examples}  """ 
    
    prompt1 = f"""  

     ### Your Task: ###  

        Fix the buggy Verilog code using the input information provided below to help you identify and resolve the bug.  Be sure to follow the instructions and the response format mentioned below to ensure accurate corrections. Use the guidelines and examples provided at the end of this prompt as hints to help you understand and correct the issues. 
    

     ### Instructions: ###  

        1. Follow the input information provided carefully to identify and fix the bug. 
        2.Follow the guidelines and examples provided at the end of this prompt to learn and understand how to spot and correct common issues. Use these examples as inspiration to identify the bugs and apply the necessary fixes. 
        3. Focus on making one or more simple changes in the lines where the bug might originate.  
        4. Ensure that the overall logic and structure of the code remain unchanged. 

    

     ### Response Format: ###  

        Provide the complete, functional Verilog code without any extra words, comments, or explanations. 


     ### Input Information: ###  

        1. After comparing the output of the correct and buggy code, a mismatch was identified between the two outputs. The elements in the code responsible for this mismatch have been identified as potential sources of the bug. These elements, which may include input ports, output ports, registers, wires, or data types, are: <<<{tmp_mismatch_set}>>> 

        2. The buggy Verilog code that needs to be fixed:  

        <<< {buggy_verilog_code} >>> 


     ### Repair Guidelines and Examples: ### 

            {few_shot_examples}

        """ 
    
    
    

    print("prompt =")
    print(prompt1)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt1)) 
    #return prompt
    return prompt1


def extract_fixed_code_from_json(json_text):
    try:
        data = json.loads(json_text)
        if 'fixed_code' in data:
            return data['fixed_code']
        else:
            print("Error: 'fixed_code' key not found in the JSON data.")
            return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return None

def scenario_tech1(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")

    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time,_,_ = calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_tech1'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code}\n"
    prompt1 = f"Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: {bug_description}, the bug might be originating from one or more of the elements listed in this mismatch list:\n{tmp_mismatch_set}\n, fix the following buggy Verilog code:\n{buggy_verilog_code}\n"
    prompt2=f"""
        Your task is to correct the specified bug in the given Verilog code while maintaining its functionality. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained, which is <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list. 
        It is important to provide the complete functioning code without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>> Your task is to identify and implement the necessary correction(s) to resolve the bug while adhering to the specified guidelines. 
            """
    prompt3=f""" Your task is to provide the complete functioning code as a hardware engineer without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained in the following list <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list.  
        The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>>  """
    
    prompt4= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        3. A list of potential elements where the bug might originate. 

        Instructions: 
        ----------------
        Think step by step. Focus on making one or more simple changes within the code, targeting the lines where the bug could originate.
        Provide reasoning for every code statement you generate. Ensure not to change the overall logic and structure of the code
        Finally, return the complete, functional code between the following pattern: start_code and end_code, ensuring there are no additional words, comments, or explanations.

        



        Bug Description: 
        ----------------

        '{bug_description}' 

        Identified Differences: 
        ---------------- 

        After comparing the output of the correct and buggy code, we have identified elements that may be contributing to the bug. These elements, including Input Ports, Output Ports, Registers, Wires, or Data Types, will be listed below: 

        {tmp_mismatch_set} 
        

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    
    prompt5= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        
        Instructions: 
        ----------------
        Think step by step. Focus on making one or more simple changes within the code, targeting the lines where the bug could originate.
        Provide reasoning for every code statement you generate. Ensure not to change the overall logic and structure of the code
        Finally, return the complete, functional code between the following pattern: start_code and end_code, ensuring there are no additional words, comments, or explanations.


        Bug Description: 
        ----------------

        '{bug_description}' 

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    

    print("prompt =")
    print(prompt5)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt5)) 
    #return prompt
    return prompt5



def scenario_tech2(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")

    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time,_,_ = calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_tech1'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code}\n"
    prompt1 = f"Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: {bug_description}, the bug might be originating from one or more of the elements listed in this mismatch list:\n{tmp_mismatch_set}\n, fix the following buggy Verilog code:\n{buggy_verilog_code}\n"
    prompt2=f"""
        Your task is to correct the specified bug in the given Verilog code while maintaining its functionality. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained, which is <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list. 
        It is important to provide the complete functioning code without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>> Your task is to identify and implement the necessary correction(s) to resolve the bug while adhering to the specified guidelines. 
            """
    prompt3=f""" Your task is to provide the complete functioning code as a hardware engineer without adding extra words, comments, or explanations to your response. Your goal is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should not alter the overall logic of the code. The description of the bug is <<< {bug_description}>>>. After comparing the output of the correct and buggy code, a list of signals or variables or registers causing the bug is obtained in the following list <<<{tmp_mismatch_set}>>>. This list signifies the mismatch between the correct and buggy code, suggesting that the bug may be due to one or more elements in this list.  
        The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>>  """
    
    prompt4= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        3. A list of potential elements where the bug might originate. 

        Instructions: 
        ----------------
        
        Start by listing the lines of code from the provided mismatch list that might be causing errors. Then, analyze which of these lines need to be corrected.
        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. Ensure not to change the overall logic and structure of the code.
        Deliver a complete, functional code without additional words, comments, or explanations. Return the complete, corrected code between the following pattern: start_code and end_code.

        



        Bug Description: 
        ----------------

        '{bug_description}' 

        Identified Differences: 
        ---------------- 

        After comparing the output of the correct and buggy code, we have identified elements that may be contributing to the bug. These elements, including Input Ports, Output Ports, Registers, Wires, or Data Types, will be listed below: 

        {tmp_mismatch_set} 
        

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    

    print("prompt =")
    print(prompt4)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt4)) 
    #return prompt
    return prompt4




def scenario_tech3(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")

    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time,_,_ = calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    basic_path = f_name+'_prompt_tech3'
    output_file_path = os.path.join(directory_path, basic_path)
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    #prompt= f" fix the following buggy verilog code: \n {buggy_verilog_code}\n. generate only code without any extra words or explanation"   
    
    #prompt= f" Generate a fix for the following buggy verilog code without generating any extra words or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Generate a fix for the following buggy verilog code without generating any comments or explanation : \n {buggy_verilog_code}\n"
    #prompt= f" Give me the full working code only without generating any extra words or comments or explanation to your answer, Generate a fix for the following buggy verilog code : \n {buggy_verilog_code}\n"
    prompt1_old= f"""
        Provide a complete, functional Verilog code without adding extra words, comments, or explanations. Your task is to fix the buggy Verilog code provided below based on the bug description: {bug_description}. Make one or more simple changes within the code, targeting the lines where the bug could originate. Do not change the overall logic and structure of the code. Return the corrected code between the following pattern: start_code and end_code. 
        Follow these repair guidelines based on the bug description: If there's a defect in conditional statements, consider inverting the condition of the code block. If the issue lies in the sensitivity list, trigger an always block on the following: Signal's falling edge or signal's rising edge or any change to a variable within the block or Signal's level change. 
        When dealing with assignment block defects, consider changing a blocking assignment to nonblocking or converting a non-blocking assignment to blocking. For numeric value discrepancies, adjust the value of an identifier by either incrementing or decrementing it by 1. 
               """
    

    prompt1=f""" Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: <<<{bug_description}>>>, fix the following buggy Verilog code:\n<<<{buggy_verilog_code}>>>\n 
                Follow these repair guidelines based on the bug description: If the issue is with conditional statements, try flipping the condition. If it's about sensitivity, make sure to set triggers for when the signal goes up or down, when any variable changes, or when the signal's level changes. For assignment problems, switch between blocking and non-blocking assignments. And if there's a number problem, adjust the value by adding or subtracting 1 """
    
    prompt2=f"""
            Your task is to correct the specified bug in the given Verilog code while maintaining its functionality. The description of the bug is <<< {bug_description}>>>. To accomplish this, deliver a complete, functional code without additional words, comments, or explanations. Focus on making one or more simple changes within the code, targeting the lines where the bug might originate. However, it's crucial not to change the overall logic and structure of the code. The provided buggy Verilog code is as follows: <<< \n{buggy_verilog_code}\n >>> Your goal is to identify and implement the necessary correction(s) to resolve the bug while adhering to the specified guidelines. 
            Return the complete, corrected code between the following pattern: start_code and end_code. Follow these repair guidelines based on the bug description: 
            If there's a defect in conditional statements, consider inverting the condition of the code block. 
            If the issue lies in the sensitivity list, trigger an always block on the following: Signal's falling edge, signal's rising edge, any change to a variable within the block, or signal's level change. 
            When dealing with assignment block defects, consider changing a blocking assignment to nonblocking or converting a non-blocking assignment to blocking. 
            For numeric value discrepancies, adjust the value of an identifier by either incrementing or decrementing it by 1. 

            """
    
    prompt3=f""" Your task as a hardware engineer assistant is to provide the complete functioning Verilog code, addressing a specific bug without adding extra words, comments, or explanations to your response. Your objective is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should refrain from altering the overall logic of the code. Ensure the corrected code is returned between the following pattern: start_code and end_code. 
                Follow these repair guidelines based on the bug description: 
                If there's a defect in conditional statements, consider inverting the condition of the code block. 
                If the issue lies in the sensitivity list, verify that the always block triggers appropriately. This includes triggering the always block on the following conditions: when the signal falls or rises, when there is any change to a variable within the block, or when the signal's level changes. 
                When dealing with assignment block defects, consider changing a blocking assignment to nonblocking or converting a non-blocking assignment to blocking. 
                For numeric value discrepancies, adjust the value of an identifier by either incrementing or decrementing it by 1. 
                The bug description is as follows: <<< {bug_description} >>>. Below is the provided buggy Verilog code:  <<< \n{buggy_verilog_code}\n >>> """
    
    
    prompt3_old=f""" Your task as a hardware engineer assistant is to provide the complete functioning Verilog code, addressing a specific bug without adding extra words, comments, or explanations to your response. Your objective is to make one or more simple changes in the code, focusing on the lines where the bug might originate. However, you should refrain from altering the overall logic of the code. The bug description is as follows: <<< {bug_description} >>>. Below is the provided buggy Verilog code:  <<< \n{buggy_verilog_code}\n >>>
                Ensure the corrected code is returned between the following pattern: start_code and end_code. Follow these repair guidelines based on the bug description: 
                If there's a defect in conditional statements, consider inverting the condition of the code block. 
                If the issue lies in the sensitivity list, verify that the always block triggers appropriately. This includes triggering the always block on the following conditions: when the signal falls or rises, when there is any change to a variable within the block, or when the signal's level changes. 
                When dealing with assignment block defects, consider changing a blocking assignment to nonblocking or converting a non-blocking assignment to blocking. 
                For numeric value discrepancies, adjust the value of an identifier by either incrementing or decrementing it by 1.  """
    
    '''
    Verilog Repair Guideline:
    1. Conditional Statements:
    If there's a defect in conditional statements, invert the condition of the code block.
    2. Sensitivity List:
    If the issue lies in the sensitivity list:
    Trigger an always block on:
    Signal's falling edge.
    Signal's rising edge.
    Any change to a variable within the block.
    Signal's level change.
    3. Assignment Block:
    When dealing with assignment block defects:
    Change a blocking assignment to nonblocking.
    Convert a non-blocking assignment to blocking.
    4. Numeric Value:
    For numeric value discrepancies:
    Increment the value of an identifier by 1.
    Decrement the value of an identifier by 1.
    5. Additional Transformations:
    Invert equality.
    Invert inequality.
    Invert ULNOT.
    Switch nonblocking assignments to blocking.
    Switch blocking assignments to nonblocking.
    Change sensitivity to negative edge.
    Change sensitivity to positive edge.
    Change sensitivity to level.
    Change sensitivity to all.
    Following these steps systematically will help in identifying and rectifying defects in Verilog code efficiently.
    '''
    '''
     Follow the following guidelines for repair: 
        if the defect is in conditional statements then negate the condition of code block 
        if the defect is in sensitivity list, consider one of the following solutions:
        Trigger an always block on a signals falling edge, Trigger an always block on a signals rising edge, Trigger an always block on any change to a variable within the block, Trigger an always block when a signal is level
        if the defect in assignment block, consider the following:
        Change a blocking assignment to nonblocking, Change a non-blocking assignment to blocking 
        if defect in numeric value, perform the following:
        Increment the value of an identifier by 1, Decrement the value of an identifier by 1
    '''


    prompt4_old= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 
        3. A list of potential elements where the bug might originate. 

        Instructions: 
        ----------------
        Deliver a complete, functional code without additional words, comments, or explanations. 
        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. 
        Do not change the overall logic and structure of the code.
        Return the complete, corrected code between the following pattern: start_code and end_code.
        Based on the bug description follow the following guidelines for repair: 
        1. Conditional Statements:
        If there's a defect in conditional statements, invert the condition of the code block.
        2. Sensitivity List:
        If the issue lies in the sensitivity list:
        Trigger an always block on:
        Signal's falling edge.
        Signal's rising edge.
        Any change to a variable within the block.
        Signal's level change.
        3. Assignment Block:
        When dealing with assignment block defects:
        Change a blocking assignment to nonblocking.
        Convert a non-blocking assignment to blocking.
        4. Numeric Value:
        For numeric value discrepancies:
        Increment the value of an identifier by 1.
        Decrement the value of an identifier by 1.



        Bug Description: 
        ----------------

        '{bug_description}' 

        Identified Differences: 
        ---------------- 

        After comparing the output of the correct and buggy code, we have identified elements that may be contributing to the bug. These elements, including Input Ports, Output Ports, Registers, Wires, or Data Types, will be listed below: 

        {tmp_mismatch_set} 
        

        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """ 
    
    prompt4= f""" 

        Your Task: 
        ----------------
        Correct a buggy Verilog code provided below. You'll be given: 

        1. The buggy Verilog code. 
        2. A description of the bug. 

        Instructions: 
        ----------------
        Deliver a complete, functional code without additional words, comments, or explanations. 
        Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. 
        Do not change the overall logic and structure of the code.
        Return the complete, corrected code between the following pattern: start_code and end_code.
        Based on the bug description follow the following guidelines for repair: 
        1. Conditional Statements:
        If there's a defect in conditional statements, invert the condition of the code block.
        2. Sensitivity List:
        If the issue lies in the sensitivity list, verify that the always block triggers appropriately. This includes triggering the always block on the following conditions:
        Signal's falling edge or Signal's rising edge or any change to a variable within the block or Signal's level change.
        3. Assignment Block:
        When dealing with assignment block defects:
        Change a blocking assignment to nonblocking or Convert a non-blocking assignment to blocking.
        4. Numeric Value:
        For numeric value discrepancies:
        Adjust the value of an identifier by either incrementing or decrementing it by 1



        Bug Description: 
        ----------------

        '{bug_description}' 


        Please find the provided buggy Verilog code below: 
        ---------------- 

        {buggy_verilog_code} 

        """
    
    prompt5= f"""Think step by step. Focus on making one or more simple changes within the code, targeting the lines where the bug could originate. Provide reasoning for every code statement you generate. Ensure not to change the overall logic and structure of the code.
                 Provide the complete functioning code without adding extra words, comments, or explanations to your answer. Given the bug description: <<<{bug_description}>>>, fix the following buggy Verilog code:

                <<<{buggy_verilog_code}>>>

                Follow these repair guidelines based on the bug description: If the issue is with conditional statements, try flipping the condition. If it's about sensitivity, make sure to set triggers for when the signal goes up or down, when any variable changes, or when the signal's level changes. For assignment problems, switch between blocking and non-blocking assignments. And if there's a number problem, adjust the value by adding or subtracting 1.
                Lastly, ensure the corrected code is returned between the following pattern: start_code and end_code, ensuring there are no additional words, comments, or explanations.    """

    print("prompt =")
    print(prompt1)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt1)) 
    #return prompt
    return prompt1




def feedback_path_creation(file_name,directory_path,iteration):

    f_name=file_name.split('.v')[0]
    directory_path=directory_path+f_name
    

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    directory_path=directory_path+"/Run"+"_"+str(iteration)
    #directory_path = os.path.join(directory_path, "Run")
    #directory_path = os.path.join(directory_path,iteration)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    
    return directory_path
    
        




def feedback_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,iteration,feedback_logic,current_trial,max_trials):
    

    optparser = OptionParser()
    optparser.add_option("-v","--version",action="store_true",dest="showversion",
                         default=False,help="Show the version")
    optparser.add_option("-I","--include",dest="include",action="append",
                         default=[],help="Include path")
    optparser.add_option("-D",dest="define",action="append",
                         default=[],help="Macro Definition")
    (options, args) = optparser.parse_args()

    filelist = [buggy_src_file_path, test_bench_path]
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")

    for f in filelist:
        if not os.path.exists(f): raise IOError("file not found: " + f)

    codegen = ASTCodeGenerator()
    print("Reacheeeeeeeeeeeeeeeeeeeeeeeeeeeeeed")
    # parse the files (in filelist) to ASTs (PyVerilog ast)
    ast, directives = parse([buggy_src_file_path],
                            preprocess_include=PROJ_DIR.split(","),
                            preprocess_define=options.define)

    
    #ast, _ = parse([buggy_src_file_path])

    ast.show()
    print(ast)
    src_code = codegen.visit(ast)
    print(src_code)

    print("\n\n")
    
    print("file_name: ")
    print(file_name)
    print("buggy_src_file_path: ")
    print(buggy_src_file_path)
    print("directory_path: ")
    print(directory_path)
    print("PROJ_DIR: ")
    print(PROJ_DIR)

    
    if(feedback_logic!=0 and current_trial!=0):
        
        try:
            # Copy the file from source path to destination path
            dst_path = os.path.join(PROJ_DIR, file_name)
            shutil.copy(buggy_src_file_path, dst_path)
            print(f"File copied from '{buggy_src_file_path}' to '{dst_path}'.")
            print("dst_path: ")
            print(dst_path)
            print("buggy_src_file_path: ")
            print(buggy_src_file_path)
            print("directory_path: ")
            print(directory_path)
            print("PROJ_DIR: ")
            print(PROJ_DIR)
            

            # Delete the copied file
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"An error occurred: {e}")



    mutation_op = MutationOp(0, True, True)
    orig_fitness, sim_time,_,_ = calc_candidate_fitness(TB_ID,EVAL_SCRIPT, orig_file_name, file_name, PROJ_DIR,oracle_path)
    print("orig_fitness = ")
    print(orig_fitness)
    
    mismatch_set, uniq_headers = get_output_mismatch(TB_ID,oracle_path)
    print(mismatch_set)
    
    if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID)

    comp_failures = 0
    
    if mutation_op.fault_loc:
        tmp_mismatch_set = copy.deepcopy(mismatch_set)
        print()
        mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers) # compute fault localization for the parent
        print("Initial Fault Localization:", str(mutation_op.fault_loc_set))
        while len(mutation_op.new_vars_in_fault_loc) > 0:
            new_mismatch_set = set(mutation_op.new_vars_in_fault_loc.values())
            print("New vars in fault loc:", new_mismatch_set)
            mutation_op.new_vars_in_fault_loc = dict()
            tmp_mismatch_set = tmp_mismatch_set.union(new_mismatch_set)
            mutation_op.get_fault_loc_targets(ast, tmp_mismatch_set, uniq_headers)
            print("Fault Localization:", str(mutation_op.fault_loc_set))
        print("Final mismatch set:", tmp_mismatch_set)
        print("Final Fault Localization:", str(mutation_op.fault_loc_set))
        print(len(mutation_op.fault_loc_set))
        # print(mutation_op.stoplist)
        # print(mutation_op.wires_brought_in)
        
                # exit(1)

        mutation_op.implicated_lines = set()
        mutation_op.collect_lines_for_fl(ast)
        print("Lines implicated by FL: %s" % str(mutation_op.implicated_lines))
        print("Number of lines implicated by FL: %d" % len(mutation_op.implicated_lines))

    if(feedback_logic!=0 and current_trial!=0):
        try:
            os.remove(dst_path)
            print(f"File deleted from '{dst_path}'.")
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

   
    f_name=file_name.split('.v')[0]
    '''
    directory_path=directory_path+f_name
    

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    '''
    directory_path=directory_path+"/iteration"+"_"+str(current_trial)


    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
        
    directory_path=directory_path+"/prompt/"
    
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 

    #qqqqqqqqqqqqq
    basic_path = f_name+'_prompt_feedback'
    output_file_path = os.path.join(directory_path, basic_path)
    
    
    with open(buggy_src_file_path, 'r') as conf_file:
        buggy_verilog_code = conf_file.read()
    
    
    prompt2=f""" Your task is to fix the given buggy Verilog code with the smallest possible changes needed to correct any existing issues, while ensuring the rest of the code remains unaltered and fully functional. Focus on making only the essential modifications to the specific lines related to the bug, and double-check that the changes do not interfere with the overall functionality or introduce new issues. provide the complete, functioning version without adding any extra words, comments, or explanations in your response.
    After generating the code, append a distinct section at the end, labeled "Code Changes". In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.
  
    The buggy Verilog code requiring correction is as follows:  <<< \n{buggy_verilog_code}\n >>>. """
    
    prompt5= f""" Your task is to fix the given buggy Verilog code and provide the complete, functioning version without adding any extra words, comments, or explanations in your response. 

    Your goal is to make one or more simple changes, focusing on the lines where the bug might originate, while ensuring that the overall logic and structure of the code remain unchanged.
    After generating the code, append a distinct section at the end, labeled "Code Changes". In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.
  

    The buggy Verilog code that needs to be fixed is as follows:  

    <<< \n{buggy_verilog_code}\n >>>.  """ 


    prompt4_old = f"""This is an iterative prompt. You have up to {max_trials} attempts to make the best possible corrections based on feedback I will provide after each response. I would like you to learn from the feedback and make the best changes based on the task given to you.

        Your task is to fix the given buggy Verilog code with the smallest possible changes needed to correct any existing issues, ensuring the rest of the code remains unaltered and fully functional. Focus on making only one essential change at a time. In future iterations, avoid repeating changes that have not resolved the issue. If a change did not work, do not apply it again. Double-check that the changes you make do not interfere with the overall functionality or introduce new issues. Provide the complete, functioning version without adding any extra words, comments, or explanations in your response.

        After generating the code, append a distinct section at the end, labeled 'Code Changes.' In this section, outline and explain the variances between the corrected and original buggy code, specifying the exact changes made, including line numbers. Ensure this section remains commented to avoid compiler interference.

        The buggy Verilog code requiring correction is as follows:

        <<< \n{buggy_verilog_code}\n >>>."""

    prompt4_old1 = f"""This is an iterative prompt. You have up to {max_trials} attempts to make the best possible corrections based on feedback I will provide after each response. I would like you to learn from the feedback and make the best changes based on the task given to you.

    Your task is to fix the given buggy Verilog code with the smallest possible changes needed to correct any existing issues, ensuring the rest of the code remains unaltered and fully functional. Focus on making only one essential change at a time. In future iterations, avoid repeating changes that have not resolved the issue. If a change did not work, do not apply it again. Keep track of the changes you made in each iteration to avoid repeating unsuccessful fixes. Always think about the logic of the code and use any input information provided to help identify the bug and make the appropriate changes. Double-check that the changes you make do not interfere with the overall functionality or introduce new issues. Provide the complete, functioning version without adding any extra words, comments, or explanations in your response.

    After generating the code, append a distinct section at the end, labeled 'Code Changes.' In this section, outline and explain the variances between the corrected and original buggy code, specifying the exact changes made, including line numbers. Ensure this section remains commented to avoid compiler interference.

    The buggy Verilog code requiring correction is as follows:

    <<< \n{buggy_verilog_code}\n >>>."""



    prompt4= f"""This is an iterative prompt. You have up to {max_trials} attempts to make the best possible corrections based on feedback I will provide after each response. I would like you to learn from the feedback and make the best changes based on the task given to you.

        Your task is to fix the given buggy Verilog code with the smallest possible changes needed to correct any existing issues, ensuring the rest of the code remains unaltered and fully functional. Focus on making only one essential change at a time. In future iterations, avoid repeating changes that have not resolved the issue. If a change did not work, do not apply it again. Keep track of the changes you made in each iteration to avoid repeating unsuccessful fixes. If you made a change in a line in one iteration and it didn't fix the code, do not repeat it for the next iterations. Always think about the logic of the code and use any input information provided to help identify the bug and make the appropriate changes. Double-check that the changes you make do not interfere with the overall functionality or introduce new issues. Provide the complete, functioning version without adding any extra words, comments, or explanations in your response.
        After generating the code, append a distinct section at the end, labeled 'Code Changes.' In this section, outline and explain the variances between the corrected and original buggy code, specifying the exact changes made, including line numbers. Ensure this section remains commented to avoid compiler interference.

        The buggy Verilog code requiring correction is as follows:

        <<< \n{buggy_verilog_code}\n >>>."""
    
    prompt4_few= f"""This is an iterative prompt. You have up to {max_trials} attempts to make the best possible corrections based on feedback I will provide after each response. I would like you to learn from the feedback and make the best changes based on the task given to you.

        Your task is to fix the given buggy Verilog code with the smallest possible changes needed to correct any existing issues, ensuring the rest of the code remains unaltered and fully functional. Focus on making only one essential change at a time. In future iterations, avoid repeating changes that have not resolved the issue. If a change did not work, do not apply it again. Keep track of the changes you made in each iteration to avoid repeating unsuccessful fixes. If you made a change in a line in one iteration and it didn't fix the code, do not repeat it for the next iterations. Always think about the logic of the code and use any input information provided to help identify the bug and make the appropriate changes. Double-check that the changes you make do not interfere with the overall functionality or introduce new issues. Provide the complete, functioning version without adding any extra words, comments, or explanations in your response.
        After generating the code, append a distinct section at the end, labeled 'Code Changes.' In this section, outline and explain the variances between the corrected and original buggy code, specifying the exact changes made, including line numbers. Ensure this section remains commented to avoid compiler interference.

        The buggy Verilog code requiring correction is as follows:

        <<< \n{buggy_verilog_code}\n >>>.

        {few_shot_examples} """






    print("prompt =")
    print(prompt4)
    
    with open(output_file_path, 'w') as file_a:
        file_a.write(str(prompt4)) 
    
    #return prompt
    return prompt4 ,orig_fitness 

def get_first_part(content):
    # Split the content into two parts based on "Code Changes"
    parts = content.split("Code Changes", 1)
    
    # Check if there are two parts
    if len(parts) == 2:
        # Get the first part before "Code Changes"
        first_part = parts[0].strip()
    else:
        # If "Code Changes" is not found, set first_part to None
        first_part = None
    
    return first_part

def get_second_part(content):
    # Split the content into two parts based on "Code Changes"
    parts = content.split("Code Changes", 1)
    
    # Check if there are two parts
    if len(parts) == 2:
        # Get the second part after "Code Changes", stripping any leading and trailing whitespace and comments
        second_part = parts[1].strip().lstrip('/*').rstrip('*/').strip()
    else:
        # If "Code Changes" is not found, set second_part to None
        second_part = None
    
    return second_part




def split_string(content):
    # Split the content into two parts based on "Code Difference"
    parts = content.split("Code Changes", 1)

    # Check if there are two parts
    if len(parts) == 2:
        # Get the second part after "Code Difference"
        second_part = "Code Changes" + parts[1].strip()  # Add "Code Changes" to the beginning of the second part
    else:
        second_part = None  # If "Code Changes" is not found, set second_part to None

    return second_part




def get_completion_from_messages(messages):
    api_key = OPENAI_API_KEY
    #client = OpenAI(api_key=api_key)
    client = OpenAI(organization='add_your_organization',api_key=api_key) # Add if your belong to an organization
    #client.models.list()
    #model_type = "gpt-3.5-turbo-1106"
    #model_type="gpt-4"
    model_type="gpt-4o-2024-05-13"

    '''
    for item in messages[1:]:
        print(item['content'])
        num_tokens_prompt_before=num_tokens_from_string(item['content'], model_type)
    '''

    num_tokens_prompt_before = 0  # Initialize the variable to accumulate the number of tokens

    # Iterate over the list starting from the second element
    for item in messages[1:]:
        num_tokens_prompt_before += num_tokens_from_string(item['content'], model_type)
        print(item['content'])

    print("\n aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \n")
    print(messages)
    print("\n aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \n")
    print("num_tokens_prompt_before = ")
    print(num_tokens_prompt_before)
    #############################
    ##start_time
    t_start = time.time()


    response = client.chat.completions.create(
        model=model_type,
        messages=messages
         # this is the degree of randomness of the model's output
    )
    t_finish = time.time()
    
    total_time= t_finish - t_start
    ## finish_time
    ###################################
    
    num_tokens_output_after=num_tokens_from_string(str(response.choices[0].message.content), model_type)
    
    ##cost_calculation
    if model_type=="gpt-3.5-turbo-1106":
        cost_before= ((num_tokens_prompt_before)*(0.0010))/(1000)
        cost_after= ((num_tokens_output_after)*(0.0020))/(1000)
        total_cost = cost_before + cost_after
        
    if model_type=="gpt-4":
        cost_before= ((num_tokens_prompt_before)*(0.03))/(1000)
        cost_after= ((num_tokens_output_after)*(0.06))/(1000)
        total_cost = cost_before + cost_after

    if model_type=="gpt-4o-2024-05-13":

        cost_before = num_tokens_prompt_before * 0.000005
        cost_after = num_tokens_output_after * 0.000015
        total_cost = cost_before + cost_after

#     print(str(response.choices[0].message))
    #return str(response.choices[0].message.content),model_type
    return str(response.choices[0].message.content),num_tokens_prompt_before,num_tokens_output_after,cost_before,cost_after,total_cost,model_type,total_time

def collect_messages1(prompt):
    
    print("context=")
    print(context)
    context.append({'role':'user', 'content':f"{prompt}"})
    print("context=")
    print(context)
    print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time = get_completion_from_messages(context) 
    print(response)
    second_part = split_string(response)
    if second_part:
        print("Second Part (after 'Code Difference'):")
        print(second_part)
    else:
        print("No 'Code Difference' found.")

    #test_part
    #context.append({'role':'assistant', 'content':f"{response}"})
    #print("context=")
    #print(context)
    
    
    return response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time

def collect_messages2(prompt):
    
    print("context=")
    print(context)
    context.append({'role':'user', 'content':f"{prompt}"})
    print("context=")
    print(context)
    response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time = get_completion_from_messages(context) 
    print(response)
    second_part = split_string(response)
    if second_part:
        print("Second Part (after 'Code Difference'):")
        print(second_part)
    else:
        print("No 'Code Difference' found.")

    #test_part
    #context.append({'role':'assistant', 'content':f"{response}"})
    #print("context=")
    #print(context)
    
    
    return response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time

def collect_messages3(prompt):
    
    print("context=")
    print(context)
    #context.append({'role':'user', 'content':f"{prompt}"})
    #print("context=")
    #print(context)
    response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time = get_completion_from_messages(context) 
    print(response)
    second_part = split_string(response)
    if second_part:
        print("Second Part (after 'Code Difference'):")
        print(second_part)
    else:
        print("No 'Code Difference' found.")

    #test_part
    #context.append({'role':'assistant', 'content':f"{response}"})
    #print("context=")
    #print(context)
    
    
    return response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time

def collect_messages4(prompt,path_select):
    
    #print("context=")
    #print(context)
    if path_select == 1:
        context.append({'role':'user', 'content':f"{prompt}"})
        print("select path ==1")


    #print("context=")
    #print(context)
    response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time = get_completion_from_messages(context) 
    #print(response)
    second_part = split_string(response)
    if second_part:
        print("Second Part (after 'Code Difference'):")
        print(second_part)
    else:
        print("No 'Code Difference' found.")

    #test_part
    #context.append({'role':'assistant', 'content':f"{response}"})
    #print("context=")
    #print(context)
    
    
    return response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time    


def get_logprobs(prompt, model="gpt-4o-2024-05-13"):
    """
    Retrieve log probabilities for each token in the prompt.
    """
    api_key = OPENAI_API_KEY
    client = OpenAI(organization='Add_organization_number',api_key=api_key)# add your organization number (if you belong to one)
    response = client.chat.completions.create(
        model=model,
        messages=[
        {"role": "system", "content": "You are a helpful assistant for fixing Verilog and system Verilog code "},
        {"role": "user", "content": prompt}
    ],
        max_tokens=1,  # We need logprobs for the given prompt
        logprobs=True     # Request log probabilities (this may vary based on API capabilities)
    )
    # Extract log probabilities from the response
    logprobs = response.choices[0].logprobs.content
    return logprobs

def calculate_perplexity(prompt):
    """
    Calculate the perplexity of a given prompt.
    """
    # Obtain log probabilities for the prompt
    logprobs = get_logprobs(prompt)
    
    # Handle None values and calculate negative log likelihoods
    nlls = [-lp if lp is not None else -100 for lp in logprobs]
    
    # Calculate the average negative log likelihood
    average_nll = np.mean(nlls)
    
    # Calculate perplexity
    perplexity = np.exp(average_nll)
    print("logprobs")
    print(logprobs)
    return perplexity




def send_prompt_chatgpt(prompt):
    #evaluate number of tokens input and output chatgpt use chatgpt tokenizer
    #cost
    #evaluate time 
    #prompt="what is my name ?"
    
    api_key = OPENAI_API_KEY
    #client = OpenAI(api_key=api_key)
    client = OpenAI(organization='add_your_organization',api_key=api_key)
    #client.models.list()
    #model_type = "gpt-3.5-turbo-1106"
    #model_type="gpt-4"
    #model_type="gpt-4o"
    model_type="gpt-4o-2024-05-13"
    #num_tokens_prompt_before=num_tokens_from_string(prompt, model_type)
    num_tokens_prompt_before=num_tokens_from_string_gpto(prompt, model_type)
    
    #num_tokens_prompt_before=0
    #############################
    ##start_time
    t_start = time.time()
    
    completion = client.chat.completions.create(
    #model="gpt-4",
    model=model_type,
    logprobs=True,
    messages=[
        {"role": "system", "content": "You are a helpful assistant for detecting and fixing bugs in Verilog and SystemVerilog code. "},
        {"role": "user", "content": prompt}
    ]
     
    )
    t_finish = time.time()
    
    total_time= t_finish - t_start
    ## finish_time
    ###################################

    ###############calculate perplexity##################
    #prex=calculate_perplexity(prompt)
    #print("prex=")
    #print(prex)
   # logprobs = [token.logprob for token in completion.choices[0].logprobs.content]
    '''
    completion.choices[0].logprobs.content
    response_text = completion.choices[0].message.content
    response_text_tokens = [token.token for token in completion.choices[0].logprobs.content]
    max_starter_length = max(len(s) for s in ["Prompt:", "Response:", "Tokens:", "Logprobs:", "Perplexity:"])
    max_token_length = max(len(s) for s in response_text_tokens)
    formatted_response_tokens = [s.rjust(max_token_length) for s in response_text_tokens]
    formatted_lps = [f"{lp:.2f}".rjust(max_token_length) for lp in logprobs]
    perplexity_score = np.exp(-np.mean(logprobs))
    print("Prompt:".ljust(max_starter_length), prompt)
    print("Response:".ljust(max_starter_length), response_text, "\n")
    print("Tokens:".ljust(max_starter_length), " ".join(formatted_response_tokens))
    print("Logprobs:".ljust(max_starter_length), " ".join(formatted_lps))
    print("Perplexity:".ljust(max_starter_length), perplexity_score, "\n")
    '''

   # perplexity_score = np.exp(-np.mean(logprobs))
    #print(perplexity_score) # Outputs the perplexity of the text
    
    ######################################################
    
    #num_tokens_output_after=num_tokens_from_string(str(completion.choices[0].message.content), model_type)
    num_tokens_output_after=num_tokens_from_string_gpto(str(completion.choices[0].message.content), model_type)
    #num_tokens_output_after=0
    
    ##cost_calculation
    if model_type=="gpt-3.5-turbo-1106":
        cost_before= ((num_tokens_prompt_before)*(0.0010))/(1000)
        cost_after= ((num_tokens_output_after)*(0.0020))/(1000)
        total_cost = cost_before + cost_after
        
    if model_type=="gpt-4":
        cost_before= ((num_tokens_prompt_before)*(0.03))/(1000)
        cost_after= ((num_tokens_output_after)*(0.06))/(1000)
        total_cost = cost_before + cost_after

    if model_type=="gpt-4o-2024-05-13":

        cost_before = num_tokens_prompt_before * 0.000005
        cost_after = num_tokens_output_after * 0.000015
        total_cost = cost_before + cost_after

        #cost_before= ((num_tokens_prompt_before)*(0.0005))/(1000)
        #cost_after= ((num_tokens_output_after)*(0.0010))/(1000)
        #total_cost = cost_before + cost_after
    
    return str(completion.choices[0].message.content),num_tokens_prompt_before,num_tokens_output_after,cost_before,cost_after,total_cost,model_type,total_time
    

################################################################################
#change for chgatgpt4o to get extra tokens to fix the big files
def get_completion_from_messages_4o(messages):
    api_key = OPENAI_API_KEY
    #client = OpenAI(api_key=api_key)
    client = OpenAI(organization='add_your_organization',api_key=api_key)
    #client.models.list()
    #model_type = "gpt-3.5-turbo-1106"
    model_type="gpt-4o-2024-05-13"

    '''
    for item in messages[1:]:
        print(item['content'])
        num_tokens_prompt_before=num_tokens_from_string(item['content'], model_type)
    '''

    num_tokens_prompt_before = 0  # Initialize the variable to accumulate the number of tokens

    # Iterate over the list starting from the second element
    for item in messages[1:]:
        num_tokens_prompt_before += num_tokens_from_string_gpto(item['content'], model_type)
        print(item['content'])

    print("\n aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \n")
    print(messages)
    print("\n aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \n")
    print("num_tokens_prompt_before = ")
    print(num_tokens_prompt_before)
    #############################
    ##start_time
    t_start = time.time()


    response = client.chat.completions.create(
        model=model_type,
        messages=messages
         # this is the degree of randomness of the model's output
    )
    t_finish = time.time()
    
    total_time= t_finish - t_start
    ## finish_time
    ###################################
    
    num_tokens_output_after=num_tokens_from_string_gpto(str(response.choices[0].message.content), model_type)
    
    ##cost_calculation
    if model_type=="gpt-4o-2024-05-13":

        cost_before = num_tokens_prompt_before * 0.000005
        cost_after = num_tokens_output_after * 0.000015
        total_cost = cost_before + cost_after

        #cost_before= ((num_tokens_prompt_before)*(0.0005))/(1000)
        #cost_after= ((num_tokens_output_after)*(0.0010))/(1000)
        #total_cost = cost_before + cost_after

    if model_type=="gpt-3.5-turbo-1106":
        cost_before= ((num_tokens_prompt_before)*(0.0010))/(1000)
        cost_after= ((num_tokens_output_after)*(0.0020))/(1000)
        total_cost = cost_before + cost_after
        
    if model_type=="gpt-4":
        cost_before= ((num_tokens_prompt_before)*(0.03))/(1000)
        cost_after= ((num_tokens_output_after)*(0.06))/(1000)
        total_cost = cost_before + cost_after

#     print(str(response.choices[0].message))
    #return str(response.choices[0].message.content),model_type
    return str(response.choices[0].message.content),num_tokens_prompt_before,num_tokens_output_after,cost_before,cost_after,total_cost,model_type,total_time

def collect_messages_4o(context4o,prompt):
    
    print("context4o=")
    print(context4o)
    context4o.append({'role':'user', 'content':f"{prompt}"})
    print("context4o=")
    print(context4o)
    print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time = get_completion_from_messages_4o(context4o) 
    print(response)
    
    
    return response ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time


# function to generalize and use it in all functions
def small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt):
    
    context4o = [ {'role':'system', 'content':"You are a helpful assistant for detecting and fixing bugs in Verilog and SystemVerilog code."} ] 
    print("context4o=")
    print(context4o)
    context4o.append({'role':'user', 'content':f"{prompt}"})
    context4o.append({'role':'assistant', 'content':f"{output_postprocess}"})


    #prompt= "Continue generating the code "
    #prompt= "Continue generating the code exactly where it stopped"
    prompt= "Continue generating the code exactly from the character where it stopped. Ensure that the generated code is a seamless continuation, starting from the exact point of interruption, without repeating or skipping any characters. If the generation stopped mid-token or mid-comment, begin from that exact character to avoid any syntax or compilation errors."

    #prompt= "Continue generating the code exactly where it stopped, ensuring not to repeat any lines that have already been generated. If the generation stopped in the middle of a comment or a code token, start the continuation from the beginning of that comment or token, but only include it once to avoid duplication. Ensure that the generated code maintains the structure and logic of the previous code without causing any syntax or compilation errors."

    output_postprocess_temp,number_tokens_input_temp,number_tokens_output_temp,cost_input_temp,cost_output_temp,cost_total_temp,model_type,time_temp=collect_messages_4o(context4o,prompt)
    
    output_postprocess = output_postprocess + output_postprocess_temp
    number_tokens_output = number_tokens_output + number_tokens_output_temp
    number_tokens_input = number_tokens_input + number_tokens_input_temp
    cost_input = cost_input + cost_input_temp
    cost_output = cost_output + cost_output_temp
    cost_total = cost_total + cost_total_temp
    time = time + time_temp
    #print("output_postprocess:")
    #print(output_postprocess)
    print("number_tokens_output:")
    print(number_tokens_output)
    print("number_tokens_input:")
    print(number_tokens_input)
    print("cost_input:")
    print(cost_input)
    print("cost_output:")
    print(cost_output)
    print("cost_total:")
    print(cost_total)
    print("time:")
    print(time)

    return output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time








#############################################################################

def techniques_main_output(output_postprocess,iteration,scenario_ID,file_name,model_type,directory_path):
    

    f_name=file_name.split('.v')[0]
    
    directory_output= directory_path +f_name+"/techniques_main_output"
       
        
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)
        print(f"Directory '{directory_output}' created.")
    else:
        print(f"Directory '{directory_output}' already exists.")
        
        
        
    init_output_file_name_before = f_name+'_iter_'+str(iteration)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.txt'

    
    output_file_path_before = os.path.join(directory_output, init_output_file_name_before)
    
    
    with open(output_file_path_before, 'w') as file_b:
        file_b.write(str(output_postprocess))
    
        



def gpt_output_postprocessing(output_postprocess,iteration,scenario_ID,file_name,model_type,directory_path):
    

    f_name=file_name.split('.v')[0]
    
    directory_preprocessing= directory_path +f_name+"/output_preprocessing"
    directory_postprocessing= directory_path +f_name+"/output_postprocessing"
       
        
    if not os.path.exists(directory_preprocessing):
        os.makedirs(directory_preprocessing)
        print(f"Directory '{directory_preprocessing}' created.")
    else:
        print(f"Directory '{directory_preprocessing}' already exists.")
        
    if not os.path.exists(directory_postprocessing):
        os.makedirs(directory_postprocessing)
        print(f"Directory '{directory_postprocessing}' created.")
    else:
        print(f"Directory '{directory_postprocessing}' already exists.")
        
        
        
    init_output_file_name_before = f_name+'_iter_'+str(iteration)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.v'
    init_output_file_name_after = f_name+'_iter_'+str(iteration)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.v'
    
    output_file_path_before = os.path.join(directory_preprocessing, init_output_file_name_before)
    output_file_path_after = os.path.join(directory_postprocessing, init_output_file_name_after)
    
    
    with open(output_file_path_before, 'w') as file_b:
        file_b.write(str(output_postprocess))
    
    
 
    #new_string = output_postprocess.replace("```verilog", "")
    new_string = output_postprocess.replace("```verilog\n", "")
    new_string = new_string.replace("```", "")
    
    with open(output_file_path_after, 'w') as file_f:
        file_f.write(str(new_string))
        
    return init_output_file_name_after, output_file_path_after ,directory_postprocessing 


def gpt_output_postprocessing_feedback(output_postprocess,current_trial,scenario_ID,file_name,model_type,directory_path,main_prompt):
    

    f_name=file_name.split('.v')[0]
    
    directory_preprocessing= directory_path+"/iteration"+"_"+str(current_trial)+"/output_preprocessing"
    directory_postprocessing= directory_path+"/iteration"+"_"+str(current_trial)+"/output_postprocessing"
    #directory_main_prompt= directory_path+"/iteration"+"_"+str(current_trial)+"/main_prompt_send"
    
        
    if not os.path.exists(directory_preprocessing):
        os.makedirs(directory_preprocessing)
        print(f"Directory '{directory_preprocessing}' created.")
    else:
        print(f"Directory '{directory_preprocessing}' already exists.")
        
    if not os.path.exists(directory_postprocessing):
        os.makedirs(directory_postprocessing)
        print(f"Directory '{directory_postprocessing}' created.")
    else:
        print(f"Directory '{directory_postprocessing}' already exists.")

    '''
    if not os.path.exists(directory_main_prompt):
        os.makedirs(directory_main_prompt)
        print(f"Directory '{directory_main_prompt}' created.")
    else:
        print(f"Directory '{directory_main_prompt}' already exists.")
    '''   
        
        
        
    #init_output_file_name_before = f_name+'_iter_'+str(current_trial)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.v'
    #init_output_file_name_after = f_name+'_iter_'+str(current_trial)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.v'
    init_output_file_name_before = f_name+'_iter_'+str(current_trial)+'.v'
    init_output_file_name_after = f_name+'_iter_'+str(current_trial)+'.v'
    init_output_file_name_before_prompt= f_name+'_iter_'+str(current_trial)+'prompt'
    init_output_file_name_after_prompt = f_name+'_iter_'+str(current_trial)+'prompt'
    #main_prompt_output = f_name+'_iter_'+str(current_trial)+'_Sc_ID_'+str(scenario_ID)+"_model_type_"+str(model_type)+'.txt'
    
    output_file_path_before = os.path.join(directory_preprocessing, init_output_file_name_before)
    output_file_path_after = os.path.join(directory_postprocessing, init_output_file_name_after)
    output_file_path_before_prompt = os.path.join(directory_preprocessing, init_output_file_name_before_prompt)
    output_file_path_after_prompt = os.path.join(directory_postprocessing, init_output_file_name_after_prompt)
    #output_file_main_prompt_path = os.path.join(directory_main_prompt, main_prompt_output)


    with open(output_file_path_before_prompt, 'w') as file_c:
        file_c.write(str(context))

    with open(output_file_path_after_prompt, 'w') as file_d:
        file_d.write(str(main_prompt))

    with open(output_file_path_before, 'w') as file_b:
        file_b.write(str(output_postprocess))
    
    #with open(output_file_main_prompt_path, 'w') as file_c:
    #    file_c.write(str(main_prompt))
    
    print(str(main_prompt))
    
    print(str(context))
    #new_string = output_postprocess.replace("```verilog", "")
    new_string = output_postprocess.replace("```verilog\n", "")
    new_string = new_string.replace("```", "")
    
    with open(output_file_path_after, 'w') as file_f:
        file_f.write(str(new_string))
      
    return init_output_file_name_after, output_file_path_after ,directory_postprocessing 


def save_output_from_gpt(file_name,scenario_ID,Iteration_number,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time):
    # create and save to pandas
    

    # Load CSV file into a pandas DataFrame
    pandas_save_results = pd.read_csv(pandas_save_result_path)
    new_row_values = [len(pandas_save_results),file_name,scenario_ID,Iteration_number,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time]  # Replace with your values
    #print(pandas_save_results)
    pandas_save_results.loc[len(pandas_save_results)] = new_row_values
    #print(pandas_save_results)
    # Step 3: Save the updated DataFrame to the CSV file
    pandas_save_results.to_csv(pandas_save_result_path, index=False)
    #filename
    #scenario_ID
    #iteration_number
    #cost input
    #cost output
    #cost_total
    #model type
    #fitness value
    #simulation pass or fail
    # fix or not
    #save time calculation
    #save token number before and after 


########################################################################

def save_output_from_gpt2_nooo(filename, scenario_ID, Iteration_number, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='output_test.csv'):
    
    folder_path ="/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/scrpit_output_4o/"
    # Ensure the folder exists

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create the full file path
    file_path = os.path.join(folder_path, output_file)

    # Check if the file exists and load it
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # If the file doesn't exist, create a new DataFrame with headers
        df = pd.DataFrame(columns=[
            "Filename", "Scenario ID", "Tokens Input", "Model Type", 
            "Tokens Output Iteration_1", "Tokens Output Iteration_2", "Tokens Output Iteration_3", "Tokens Output Iteration_4", "Tokens Output Iteration_5",
            "Fitness Iteration_1", "Fitness Iteration_2", "Fitness Iteration_3", "Fitness Iteration_4", "Fitness Iteration_5",
            "Cost Input", "Cost Output", "Cost Total", "Time", 
            "Simulation Status", "Simulation Time"
        ])

    # Check if this filename and scenario already exist in the dataframe
    existing_row = df[(df['Filename'] == filename) & 
                      (df['Scenario ID'] == scenario_ID)]

    if not existing_row.empty:
        # Update the row with new token input/output and iteration values
        row_index = existing_row.index[0]

        # Define the token output and fitness columns for the given iteration
        token_output_col = f"Tokens Output Iteration_{Iteration_number}"
        fitness_col = f"Fitness Iteration_{Iteration_number}"

        # Update the row with the new values for this iteration
        df.at[row_index, "Tokens Input"] = number_tokens_input
        df.at[row_index, token_output_col] = number_tokens_output
        df.at[row_index, fitness_col] = fitness_value
    else:
        # Append a new row with the initial values
        new_row = {
            "Filename": filename,
            "Scenario ID": scenario_ID,
            "Tokens Input": number_tokens_input,
            "Model Type": model_type,
            f"Tokens Output Iteration_1": number_tokens_output if Iteration_number == 1 else None,
            f"Tokens Output Iteration_2": number_tokens_output if Iteration_number == 2 else None,
            f"Tokens Output Iteration_3": number_tokens_output if Iteration_number == 3 else None,
            f"Tokens Output Iteration_4": number_tokens_output if Iteration_number == 4 else None,
            f"Tokens Output Iteration_5": number_tokens_output if Iteration_number == 5 else None,
            f"Fitness Iteration_1": fitness_value if Iteration_number == 1 else None,
            f"Fitness Iteration_2": fitness_value if Iteration_number == 2 else None,
            f"Fitness Iteration_3": fitness_value if Iteration_number == 3 else None,
            f"Fitness Iteration_4": fitness_value if Iteration_number == 4 else None,
            f"Fitness Iteration_5": fitness_value if Iteration_number == 5 else None
            #"Cost Input": cost_input,
            #"Cost Output": cost_output,
            #"Cost Total": cost_total,
            #"Model Type": model_type,
            #"Time": time,
            #"Simulation Status": simulation_status,
            #"Simulation Time": simulation_time
        }
        df = pd.concat([df, new_row], ignore_index=True)

    # Save back to the CSV file in the given folder path
    df.to_csv(file_path, index=False)


def save_output_from_gpt2(filename, scenario_ID, Iteration_number, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='output_test.csv'):
   
    folder_path ="/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/scrpit_output_4o/"
   
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create the full file path
    file_path = os.path.join(folder_path, output_file)

    # Check if the file exists and load it
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # If the file doesn't exist, create a new DataFrame with headers
        df = pd.DataFrame(columns=[
            "Filename", "Scenario ID", "Tokens Input", "Model Type", 
            "Tokens Output Iteration_1", "Tokens Output Iteration_2", "Tokens Output Iteration_3", "Tokens Output Iteration_4", "Tokens Output Iteration_5",
            "Fitness Iteration_1", "Fitness Iteration_2", "Fitness Iteration_3", "Fitness Iteration_4", "Fitness Iteration_5",
            "Cost Input", "Cost Output", "Cost Total", "Time", 
            "Simulation Status", "Simulation Time"
        ])

    # Check if this filename and scenario already exist in the dataframe
    existing_row = df[(df['Filename'] == filename) & 
                      (df['Scenario ID'] == scenario_ID)]

    if not existing_row.empty:
        # Update the row with new token input/output and iteration values
        row_index = existing_row.index[0]

        # Define the token output and fitness columns for the given iteration
        token_output_col = f"Tokens Output Iteration_{Iteration_number}"
        fitness_col = f"Fitness Iteration_{Iteration_number}"

        # Update the row with the new values for this iteration
        df.at[row_index, "Tokens Input"] = number_tokens_input
        df.at[row_index, token_output_col] = number_tokens_output
        df.at[row_index, fitness_col] = fitness_value
    else:
        # Create a new row with the initial values
        new_row = pd.DataFrame({
            "Filename": [filename],
            "Scenario ID": [scenario_ID],
            "Tokens Input": [number_tokens_input],
            "Model Type": [model_type],
            f"Tokens Output Iteration_1": [number_tokens_output if Iteration_number == 1 else None],
            f"Tokens Output Iteration_2": [number_tokens_output if Iteration_number == 2 else None],
            f"Tokens Output Iteration_3": [number_tokens_output if Iteration_number == 3 else None],
            f"Tokens Output Iteration_4": [number_tokens_output if Iteration_number == 4 else None],
            f"Tokens Output Iteration_5": [number_tokens_output if Iteration_number == 5 else None],
            f"Fitness Iteration_1": [fitness_value if Iteration_number == 1 else None],
            f"Fitness Iteration_2": [fitness_value if Iteration_number == 2 else None],
            f"Fitness Iteration_3": [fitness_value if Iteration_number == 3 else None],
            f"Fitness Iteration_4": [fitness_value if Iteration_number == 4 else None],
            f"Fitness Iteration_5": [fitness_value if Iteration_number == 5 else None]
            #"Cost Input": [cost_input],
            #"Cost Output": [cost_output],
            #"Cost Total": [cost_total],
            #"Time": [time],
            #"Simulation Status": [simulation_status],
            #"Simulation Time": [simulation_time]
        })

        # Concatenate the new DataFrame with the original one
        df = pd.concat([df, new_row], ignore_index=True)

    # Save back to the CSV file in the given folder path
    df.to_csv(file_path, index=False)

#######################################################################################

def save_output_from_gpt_feedback(filename, scenario_ID, Iteration_number, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='output_test.csv'):
   
    folder_path ="/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/scrpit_output_4o/"
   
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create the full file path
    file_path = os.path.join(folder_path, output_file)

    # Check if the file exists and load it
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # If the file doesn't exist, create a new DataFrame with headers
        df = pd.DataFrame(columns=[
            "Filename", "Scenario ID", "Tokens Input", "Model Type", 
            "Tokens Output Iteration_1", "Tokens Output Iteration_2", "Tokens Output Iteration_3", "Tokens Output Iteration_4", "Tokens Output Iteration_5",
            "Fitness Iteration_1", "Fitness Iteration_2", "Fitness Iteration_3", "Fitness Iteration_4", "Fitness Iteration_5",
            "Cost Input", "Cost Output", "Cost Total", "Time", 
            "Simulation Status", "Simulation Time"
        ])

    # Check if this filename and scenario already exist in the dataframe
    existing_row = df[(df['Filename'] == filename) & 
                      (df['Scenario ID'] == scenario_ID)]

    if not existing_row.empty:
        # Update the row with new token input/output and iteration values
        row_index = existing_row.index[0]

        # Define the token output and fitness columns for the given iteration
        token_output_col = f"Tokens Output Iteration_{Iteration_number}"
        fitness_col = f"Fitness Iteration_{Iteration_number}"

        # Update the row with the new values for this iteration
        df.at[row_index, "Tokens Input"] = number_tokens_input
        df.at[row_index, token_output_col] = number_tokens_output
        df.at[row_index, fitness_col] = fitness_value
    else:
        # Create a new row with the initial values
        new_row = pd.DataFrame({
            "Filename": [filename],
            "Scenario ID": [scenario_ID],
            "Tokens Input": [number_tokens_input],
            "Model Type": [model_type],
            f"Tokens Output Iteration_1": [number_tokens_output if Iteration_number == 1 else None],
            f"Tokens Output Iteration_2": [number_tokens_output if Iteration_number == 2 else None],
            f"Tokens Output Iteration_3": [number_tokens_output if Iteration_number == 3 else None],
            f"Tokens Output Iteration_4": [number_tokens_output if Iteration_number == 4 else None],
            f"Tokens Output Iteration_5": [number_tokens_output if Iteration_number == 5 else None],
            f"Fitness Iteration_1": [fitness_value if Iteration_number == 1 else None],
            f"Fitness Iteration_2": [fitness_value if Iteration_number == 2 else None],
            f"Fitness Iteration_3": [fitness_value if Iteration_number == 3 else None],
            f"Fitness Iteration_4": [fitness_value if Iteration_number == 4 else None],
            f"Fitness Iteration_5": [fitness_value if Iteration_number == 5 else None]
            #"Cost Input": [cost_input],
            #"Cost Output": [cost_output],
            #"Cost Total": [cost_total],
            #"Time": [time],
            #"Simulation Status": [simulation_status],
            #"Simulation Time": [simulation_time]
        })

        # Concatenate the new DataFrame with the original one
        df = pd.concat([df, new_row], ignore_index=True)

    # Save back to the CSV file in the given folder path
    df.to_csv(file_path, index=False)


#######################################################################################

def save_output_from_gpt2_old(filename,scenario_ID,Iteration_number,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time, folder_path, output_file='output_test.csv'):
    # Ensure the folder exists, if not, create it


    folder_path ="/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/scrpit_output_4o/"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create the full file path
    file_path = os.path.join(folder_path, output_file)

    # Check if the file exists and load it
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # If the file doesn't exist, create a new DataFrame with headers
        df = pd.DataFrame(columns=["Filename", "Scenario ID", "Tokens Input","model type","Tokens Output Iteration_1","Tokens Output Iteration_2","Tokens Output Iteration_3","Tokens Output Iteration_4","Tokens Output Iteration_5" ])

    # Check if this filename and scenario already exist in the dataframe
    existing_row = df[(df['Filename'] == filename) & 
                      (df['Scenario ID'] == scenario_ID)]

    if not existing_row.empty:
        # Update the row with new token input/output and iteration values
        row_index = existing_row.index[0]
        #token_input_col = f"Tokens Input Iteration {Iteration_number}"
        #token_output_col = f"Tokens Output Iteration {Iteration_number}"
        iteration_col = f"Iteration {Iteration_number}"

        # Dynamically add columns for tokens and iteration if they don't exist
        '''
        if token_input_col not in df.columns:
            df[token_input_col] = None
        if token_output_col not in df.columns:
            df[token_output_col] = None
         '''
        if iteration_col not in df.columns:
            df[iteration_col] = None

        # Update the row with the new values for this iteration
        #df.at[row_index, token_input_col] = number_tokens_input
        #df.at[row_index, token_output_col] = number_tokens_output
        df.at[row_index, iteration_col] = fitness_value
    else:
        # Append a new row with the initial values
        new_row = {
            "Filename": filename, 
            "Scenario ID": scenario_ID, 
            #f"Tokens Input Iteration {Iteration_number}": number_tokens_input,
            #f"Tokens Output Iteration {Iteration_number}": number_tokens_output,
            f"Iteration {Iteration_number}": fitness_value
        }
        df = df.append(new_row, ignore_index=True)

    # Save back to the CSV file in the given folder path
    df.to_csv(file_path, index=False)






def save_feedback_output_from_gpt(file_name,scenario_ID,run_number,Iteration_number,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time):
    # create and save to pandas
    # Load CSV file into a pandas DataFrame
    pandas_save_results = pd.read_csv(pandas_save_feedback_result_path)
    new_row_values = [len(pandas_save_results),file_name,scenario_ID,run_number,Iteration_number,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time]  # Replace with your values
    #print(pandas_save_results)
    pandas_save_results.loc[len(pandas_save_results)] = new_row_values
    #print(pandas_save_results)
    # Step 3: Save the updated DataFrame to the CSV file
    pandas_save_results.to_csv(pandas_save_feedback_result_path, index=False)




def run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory):

    print("Running VCS simulation")
    #os.system("cat %s" % fileName)
    t_start = time.time()
    
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")
    # get the filename only if full path specified
    #if "/" in output_file_path_after: output_file_path_after = output_file_path_after.split("/")[-1]

    try:
        # Extract the filename from the original path
        filename = os.path.basename(output_file_path_after)
        print("output_file_path_after=")
        print(output_file_path_after)
        print("filename=")
        print(filename)
        # Create the new path by combining the new directory and the original filename
        new_path = os.path.join(PROJ_DIR, filename)

        # Copy the file from the original path to the new path
        shutil.copy(output_file_path_after, new_path)

        # Run the bash script using the copied file in the new path
        cmd = ["bash", EVAL_SCRIPT, orig_file_name, filename, PROJ_DIR]
        process = subprocess.Popen(cmd)
        process.wait() 

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Delete the copied file after running the script
        try:
        
            os.remove(new_path)
            print(f"Successfully deleted the copied file: {new_path}")
        except Exception as e:
            print(f"Error deleting the copied file: {e}")
    # TODO: The test bench is currently hard coded in eval_script. Do we want to change that?
    #for_testing
    #os.system("bash %s %s %s %s" % (EVAL_SCRIPT, ORIG_FILE, fileName, PROJ_DIR))
    
    #might be an answer # check it first 
    
    # Construct the new file name and path
    new_file_name = f"{output_file_name_after}_{TB_ID}_output.txt"
    new_file_path = os.path.join(output_file_path_after_directory, new_file_name)


    #t_start = time.time()
    #cmd = ["bash", EVAL_SCRIPT, orig_file_name, output_file_path_after, PROJ_DIR]
    #process = subprocess.Popen(cmd)
    #process.wait() 
    
    
    if not os.path.exists("output_%s.txt" % TB_ID): 
        t_finish = time.time()
        return 0,False, t_finish - t_start # if the code does not compile, return 0
        # return math.inf

    f = open(oracle_path, "r")
    oracle_lines = f.readlines()
    f.close()
    
    # Rename and move the output file
    # de 7eta zyada
    #me7tag a test bel fitness 1 we m7tag an2el el file lel path el sa7
    #os.rename("output_%s.txt" % TB_ID, new_file_path)

    f = open("output_%s.txt" % TB_ID, "r")
    sim_lines = f.readlines()
    f.close()
    
        # Get the current working directory (where the Python script is located)
    current_directory = os.getcwd()

    # Specify the filename of the file you want to move
    file_to_move = "output_%s.txt" % TB_ID
    #new_file_name = f"{file_to_move}_iter{iteration}"

    # Specify the destination path
    destination_path = output_file_path_after_directory+"/output_simulation"

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Directory '{destination_path}' created.")
    else:
        print(f"Directory '{destination_path}' already exists.")

    try:
        # Construct the full path to the file
        original_file_path = os.path.join(current_directory, file_to_move)
        # Extract the filename without the path and extension
        original_filename, original_extension = os.path.splitext(os.path.basename(original_file_path))

    # Add the iteration before the extension
        new_filename = f"{original_filename}_iter{iteration}{original_extension}"

        # Move the file to the specified destination path
        shutil.move(original_file_path, os.path.join(destination_path, new_filename))
        print(f"Successfully moved the file to: {destination_path}")

    except Exception as e:
        print(f"An error occurred: {e}")





    # 2amove el file aw a write fe path el ana 3ayzo we a3melo remove 
    # aw momken a remove delwa2ty el output 

    #remove after testing and getting oracle file
    
    #ff, total_possible = fitness.calculate_fitness(oracle_lines, sim_lines, None, "")
    #ff = 0
    #total_possible =1
    
    #end of added code 
    #print("oracle_lines: ")
    #print(oracle_lines)
    '''
    qqqqqqsssss
    temp_path1= output_file_path_after_directory+"/sim_lines.txt"
    temp_path2= output_file_path_after_directory+"/oracle_lines.txt"
    with open(temp_path1, 'w') as file_f:
            for line in sim_lines:
                file_f.write(line + '\n')

    with open(temp_path2, 'w') as file_f:
            for line in oracle_lines:
                file_f.write(line + '\n')
    '''
    

    ff, total_possible = fitness.calculate_fitness(oracle_lines, sim_lines, None, "")
        
    normalized_ff = ff/total_possible
    if normalized_ff < 0: normalized_ff = 0
    print("FITNESS = %f" % normalized_ff)
    t_finish = time.time()
    # if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID) # Do we need to do this here? Does it make a difference?
    #t_finish = time.time()

        


   

     
    fitness_value=normalized_ff
    simulation_status=True
    simulation_time = t_finish - t_start
    fix_status=0
    #return normalized_ff, t_finish - t_start
    return fitness_value,simulation_status,simulation_time
    




def run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_code,file_name):

    print("Running VCS simulation")
    #os.system("cat %s" % fileName)
    t_start = time.time()
    
    TB_ID = test_bench_path.split("/")[-1].replace(".v","")
    file_name_without_extension = os.path.splitext(file_name)[0]
    # get the filename only if full path specified
    #if "/" in output_file_path_after: output_file_path_after = output_file_path_after.split("/")[-1]

    try:
        # Extract the filename from the original path
        filename = os.path.basename(buggy_code)

        # Create the new path by combining the new directory and the original filename
        #new_path = os.path.join(PROJ_DIR, filename)

        # Copy the file from the original path to the new path
        #shutil.copy(output_file_path_after, new_path)

        # Run the bash script using the copied file in the new path
        cmd = ["bash", EVAL_SCRIPT, orig_file_name, filename, PROJ_DIR]
        #cmd = ["bash", EVAL_SCRIPT, orig_file_name, orig_file_name, PROJ_DIR]
        process = subprocess.Popen(cmd)
        process.wait() 

    except Exception as e:
        print(f"An error occurred: {e}")

    
 
    
    if not os.path.exists("output_%s.txt" % TB_ID): 
        t_finish = time.time()
        return 0,False, t_finish - t_start # if the code does not compile, return 0
        # return math.inf

    f = open(oracle_path, "r")
    oracle_lines = f.readlines()
    f.close()
    
    # Rename and move the output file
    # de 7eta zyada
    #me7tag a test bel fitness 1 we m7tag an2el el file lel path el sa7
    #os.rename("output_%s.txt" % TB_ID, new_file_path)

    f = open("output_%s.txt" % TB_ID, "r")
    sim_lines = f.readlines()
    f.close()
    
        # Get the current working directory (where the Python script is located)
    current_directory = os.getcwd()

    # Specify the filename of the file you want to move
    file_to_move = "output_%s.txt" % TB_ID
    #new_file_name = f"{file_to_move}_iter{iteration}"

    # Specify the destination path
    destination_path = current_directory+"/baseline_output_simulation_correct"

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"Directory '{destination_path}' created.")
    else:
        print(f"Directory '{destination_path}' already exists.")

    try:
        # Construct the full path to the file
        original_file_path = os.path.join(current_directory, file_to_move)
        # Extract the filename without the path and extension
        original_filename, original_extension = os.path.splitext(os.path.basename(original_file_path))

    # Add the iteration before the extension
        new_filename = f"{file_name_without_extension}{original_extension}"
        #new_filename = f"{original_filename}{original_extension}"

        # Move the file to the specified destination path
        shutil.move(original_file_path, os.path.join(destination_path, new_filename))
        print(f"Successfully moved the file to: {destination_path}")

    except Exception as e:
        print(f"An error occurred: {e}")





    # 2amove el file aw a write fe path el ana 3ayzo we a3melo remove 
    # aw momken a remove delwa2ty el output 
    
    ff, total_possible = fitness.calculate_fitness(oracle_lines, sim_lines, None, "")
        
    normalized_ff = ff/total_possible
    if normalized_ff < 0: normalized_ff = 0
    print("FITNESS = %f" % normalized_ff)
    t_finish = time.time()
    # if os.path.exists("output_%s.txt" % TB_ID): os.remove("output_%s.txt" % TB_ID) # Do we need to do this here? Does it make a difference?
    #t_finish = time.time()

        


   

     
    fitness_value=normalized_ff
    simulation_status=True
    simulation_time = t_finish - t_start
    fix_status=0
    #return normalized_ff, t_finish - t_start
    return fitness_value,simulation_status,simulation_time
    

def save_baseline_output(file_name,fitness_value,simulation_status,simulation_time):
    # create and save to pandas
    

    # Load CSV file into a pandas DataFrame
    pandas_save_results = pd.read_csv(baseline_save_results)
    new_row_values = [len(pandas_save_results),file_name,fitness_value,simulation_status,simulation_time]  # Replace with your values
    #print(pandas_save_results)
    pandas_save_results.loc[len(pandas_save_results)] = new_row_values
    #print(pandas_save_results)
    # Step 3: Save the updated DataFrame to the CSV file
    pandas_save_results.to_csv(baseline_save_results, index=False)
    #filename
    #scenario_ID
    #iteration_number
    #cost input
    #cost output
    #cost_total
    #model type
    #fitness value
    #simulation pass or fail
    # fix or not
    #save time calculation
    #save token number before and after 





def main(args):
    global context 
    file_path = args.pandas_csv_path
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    directory_path=directory_creation(args.scenario_ID,args.experiment_number)
    start_index = 0


    if (args.scenario_ID == 0):#0= basic_scenario
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = basic_scenario(file_name,buggy_src_file_path,directory_path)
                
                for iteration in range(1, iterations + 1):
                   
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)
                    

                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                        '''
                        context4o = [ {'role':'system', 'content':"You are a helpful assistant for fixing Verilog and system Verilog code."} ] 
                        print("context4o=")
                        print(context4o)
                        context4o.append({'role':'user', 'content':f"{prompt}"})
                        context4o.append({'role':'assistant', 'content':f"{output_postprocess}"})


                        #prompt= "Continue generating the code "
                        prompt= "Continue generating the code exactly where it stopped"
                        #prompt= "Continue generating the code exactly where it stopped, ensuring not to repeat any lines that have already been generated. If the generation stopped in the middle of a comment or a code token, start the continuation from the beginning of that comment or token, but only include it once to avoid duplication. Ensure that the generated code maintains the structure and logic of the previous code without causing any syntax or compilation errors."

                        output_postprocess_temp,number_tokens_input_temp,number_tokens_output_temp,cost_input_temp,cost_output_temp,cost_total_temp,model_type,time_temp=collect_messages_4o(context4o,prompt)
                        
                        output_postprocess = output_postprocess + output_postprocess_temp
                        number_tokens_output = number_tokens_output + number_tokens_output_temp
                        number_tokens_input = number_tokens_input + number_tokens_input_temp
                        cost_input = cost_input + cost_input_temp
                        cost_output = cost_output + cost_output_temp
                        cost_total = cost_total + cost_total_temp
                        time = time + time_temp
                        print("output_postprocess:")
                        print(output_postprocess)
                        print("number_tokens_output:")
                        print(number_tokens_output)
                        print("number_tokens_input:")
                        print(number_tokens_input)
                        print("cost_input:")
                        print(cost_input)
                        print("cost_output:")
                        print(cost_output)
                        print("cost_total:")
                        print(cost_total)
                        print("time:")
                        print(time)
                        '''

                    print("output_postprocess_new:")
                    print(output_postprocess)
                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    #save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_unstructured_4o.csv')
                    save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_structured_4o.csv')
                    
                    #for getting baseline fitness
                    #save_baseline_output(file_name,fitness_value,simulation_status,simulation_time)
                    #print(output_postprocess)
                    print("/////////////////////////////////////////////////////////")
                    
                    #gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                #aaaaaaaaa
                    #gpt_output_postprocessing(file_name,iteration)
        else:
            print("wrong value")

    elif (args.scenario_ID == 20):#1= bug_description_scenario
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = bug_description_scenario(file_name,buggy_src_file_path,directory_path,bug_description)

                for iteration in range(1, iterations + 1):
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    print("output_postprocess_new:")
                    print(output_postprocess)

                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    #save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='zero_description_unstructured_4o.csv')
                    save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='zero_description_unstructured_4o.csv')
                    #save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_structured_4o.csv')
                    
                    #for getting baseline fitness
                    #save_baseline_output(file_name,fitness_value,simulation_status,simulation_time)
                    #print(output_postprocess)
                    print("/////////////////////////////////////////////////////////")
                    #gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                #aaaaaaaaa
                    #gpt_output_postprocessing(file_name,iteration)
        else:
            print("wrong value")   


    elif (args.scenario_ID == 2):#2= scenario_mismatch
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = scenario_mismatch(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)


                    # how to fix the big codes with gpt4o
                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    

                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    #save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_unstructured_4o.csv')
                    save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_structured_4o.csv')
                    
                    print("/////////////////////////////////////////////////////////")
        else:
            print("wrong value")

    elif (args.scenario_ID == 3):#3= new technique adding extra part to the prompt -->lets think step by step. give reasoning for every code statement you generate and then finally write the complete generated code 
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = scenario_tech1(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # how to fix the big codes with gpt4o
                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    
                    print("output_postprocess=")
                    print(output_postprocess)

                    techniques_main_output(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    #parsed_output= extract_fixed_code_from_json(output_postprocess)
                    #parsed_output=output_postprocess[output_postprocess.find("{"): output_postprocess.find("}")+1]

                    try:
                        parsed_output = output_postprocess.split("start_code")[1].split("end_code")[0]
                        print("parsed_output=")
                        print(parsed_output)
                        output_postprocess= parsed_output
                        #parsed1 = json.loads(parsed_output)
                        #code_fixed = parsed1['fixed_code']
                        #code_fixed = parsed_output
                    except Exception as e:
                        print(f"An error occurred: {e}")


                    #parsed_output= output_postprocess.split("start_code")[1].split("end_code")[0]
                    #print("parsed_output=")
                    #print(parsed_output)
                    #parsed1=json.loads(parsed_output)
                    #code_fixed=parsed1['fixed_code']
                    #code_fixed=parsed_output
                    print("output_postprocess=")
                    print(output_postprocess)
                    #aaaaaaaaaaaa
                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    
                    print("/////////////////////////////////////////////////////////")
        else:
            print("wrong value")
                    

    elif (args.scenario_ID == 4):#4= new technique adding extra part to the prompt -->Start by listing the lines of code from the provided mismatch list that might be causing errors. Then, analyze which of these lines need to be corrected. Finally, generate the complete corrected code incorporating the necessary fixes. 
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = scenario_tech2(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # how to fix the big codes with gpt4o
                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    
                    print("output_postprocess=")
                    print(output_postprocess)

                    techniques_main_output(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    #parsed_output= extract_fixed_code_from_json(output_postprocess)
                    #parsed_output=output_postprocess[output_postprocess.find("{"): output_postprocess.find("}")+1]

                    try:
                        parsed_output = output_postprocess.split("start_code")[1].split("end_code")[0]
                        print("parsed_output=")
                        print(parsed_output)
                        output_postprocess= parsed_output
                        #parsed1 = json.loads(parsed_output)
                        #code_fixed = parsed1['fixed_code']
                        #code_fixed = parsed_output
                    except Exception as e:
                        print(f"An error occurred: {e}")

                    #parsed_output= output_postprocess.split("start_code")[1].split("end_code")[0]
                    #print("parsed_output=")
                    #print(parsed_output)
                    #parsed1=json.loads(parsed_output)
                    #code_fixed=parsed1['fixed_code']
                    #code_fixed=parsed_output

                    
                    print("output_postprocess=")
                    print(output_postprocess)
                    #aaaaaaaaaaaa
                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    
                    print("/////////////////////////////////////////////////////////")
        else:
            print("wrong value")



    elif (args.scenario_ID == 5):#5= new technique adding extra part to the prompt --> Follow the following guidelines for repair that are mentioned in cirfix
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = scenario_tech3(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # how to fix the big codes with gpt4o
                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    
                    print("output_postprocess=")
                    print(output_postprocess)

                    techniques_main_output(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)

                    try:
                        parsed_output = output_postprocess.split("start_code")[1].split("end_code")[0]
                        print("parsed_output=")
                        print(parsed_output)
                        output_postprocess= parsed_output
                        #parsed1 = json.loads(parsed_output)
                        #code_fixed = parsed1['fixed_code']
                        #code_fixed = parsed_output
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    
                    #parsed_output= output_postprocess.split("start_code")[1].split("end_code")[0]
                    #print("parsed_output=")
                    #print(parsed_output)
                    

                    
                    print("output_postprocess=")
                    print(output_postprocess)
                    #aaaaaaaaaaaa
                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                    

                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    
                    print("/////////////////////////////////////////////////////////")

        else:
            print("wrong value")




    elif (args.scenario_ID == 88):#7= feedback scneario
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #result = df[df['file_name'] == file_name]
            #print(result)
            #prompt=basic_scenario(file_name,buggy_src_file_path,directory_path)
            
            #output_postprocess=send_prompt_chatgpt(prompt)
            #gpt_output_postprocessing(output_postprocess)
        elif(args.choose_file == "all") :
            #loop over pandas dataframe 
            #for each file in pandas dataframe column call function basicscenario
            # and for each file with this prompt run n number of iterations 
            # for each iteration send the output to chatgpt 
            # then process the output data 
            # then save the output data in a file 
            #for index, row in df.iterrows():
             # Iterate over rows starting from the specified index
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                #print("bug_description = ",bug_description)
                #print("buggy_src_file_path = ",buggy_src_file_path)
                #print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                #print("test_bench_path = ",test_bench_path)
                #print("orig_file_name = ",orig_file_name)
                #print("PROJ_DIR = ",PROJ_DIR)
                #print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations and here the iterations are run and current trial = iteration
                #prompt_initial = scenario_mismatch(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
                #prompt = feedback_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,iteration)

                #if (args.feedback_logic==1):
                for iteration in range(1, iterations + 1):

                    max_trials = 5
                    current_trial = 0

                    while current_trial < max_trials:

                        if(current_trial==0):
                            initial_context = [ {'role':'system', 'content':"You are a helpful assistant for fixing Verilog and system Verilog code."} ]
                            context = initial_context.copy()
                            iteration = iteration - 1
                            print(context)
                            directory_path_new=feedback_path_creation(file_name,directory_path,iteration)
                            main_prompt ,orig_fitness= feedback_scenario(file_name,buggy_src_file_path,directory_path_new,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,iteration,args.feedback_logic,current_trial,max_trials)
                        

                        main_response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=collect_messages1(main_prompt)
                        
                        if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                            main_response,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(main_response ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,main_prompt)
                            second_part = split_string(main_response)

                        if second_part is None:
                            second_part = ""  # Set to an empty string if it's None
                        #output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                        # azabat el input we el output fe el two functions dol
                        #replace iteration with current trials to update the files and have different files
                        output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing_feedback(main_response,current_trial,args.scenario_ID,file_name,model_type,directory_path_new,main_prompt)

                        #replace iteration with current trials to update the files and have different files
                        fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, current_trial,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                        
                        #for getting baseline fitness
                        #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                        ###########
                        #replace iteration with current trials to update the files and have different files
                        save_output_from_gpt(file_name,args.scenario_ID,current_trial,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                        save_output_from_gpt_feedback(file_name, args.scenario_ID,(current_trial+1), number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='feedback_output1_again.csv')
                        #save_feedback_output_from_gpt(file_name,args.scenario_ID,iteration,current_trial,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                        
                        print("/////////////////////////////////////////////////////////")
                        if(current_trial==0):
                        

                            if fitness_value == 1:
                                print("Optimal solution found!")
                                break

                            elif(fitness_value > orig_fitness):
                                
                                #new_fitness =fitness_value
                                #orig_fitness = fitness_value
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                                print("reacheeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed")
                                main_prompt ="\n Feedback for iteration "+str(current_trial)+": The changes you have made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time. Here are the code changes you made before:\n" +second_part
                                #main_prompt = f"\n Feedback for iteration {str(current_trial)}: The previous code changes you made did not resolve the functional issue and have worsened the code. Please review the bug and apply other changes that directly address the problem. Here are the code changes you made before:\n{second_part if second_part else ''}"

                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference. I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ", The changes you've made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time. Below are the code changes you previously suggested: \n "+second_part
                                #orig_fitness = new_fitness
                                #main_prompt = next_prompt
                                #context = [ {'role':'system', 'content':"You are a helpful assistant for fixing Verilog and system Verilog code."} ] 
                                print("context=")
                                print(context)
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                                current_trial += 1

                            #elif(fitness_value == new_fitness):
                            
                                #main_prompt="\n The changes you've made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time."
                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference. I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ", The changes you've made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time. Below are the code changes you previously suggested: \n "+second_part
                                #current_trial += 1
                            else:
                                #main_prompt = main_prompt+"\n I have send to you this code before and I wanted to reach value after simulation =1, but it is still not correct and these are the code changes you made before \n "+second_part
                                #main_prompt = main_prompt+"\n I previously sent you this code with the intention of achieving a simulation value of 1. However, despite the changes you made to the code, the simulation value is currently "+ str(fitness_value) + " these are the code changes you gave me before :\n "+second_part
                                #context[1]['content'] += "\n I previously sent you this code with the intention of achieving a simulation value of 1. However, despite the changes you made to the code, the simulation value is currently "+ str(fitness_value) + "  I want to receive different code changes as these changes did not yield the desired outcome. these are the code changes you gave me before :\n "+second_part
                                #print("context_new :")
                                #print(context)
                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + "and the response was : "+ main_response
                            

                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ". Below are the code changes you previously suggested: \n "+second_part
                                #main_prompt = "\n Feedback for iteration "+str(current_trial)+": The previous code changes you made did not resolve the functional issue and have worsened the code. Please review the bug and apply other changes that directly address the problem. If you made a change to a line in the previous iteration and it didn't improve the code, do not repeat it in subsequent iterations; instead, make changes to other lines. Here are the code changes you made before:\n" + second_part
                                main_prompt = "\n Feedback for iteration "+str(current_trial)+": The previous code changes you made did not resolve the functional issue and have worsened the code. Please review the bug and apply other changes that directly address the problem. Here are the code changes you made before:\n" + second_part


                                #main_prompt =context+"\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at 0. try to Fix the code again "
                                #main_prompt =context
                                
                                print("Fitness value is not 1. Retrying...")
                                current_trial += 1

                        elif(current_trial==1):
                        
                            prompt_final =all_input_feedback(file_name,buggy_src_file_path,directory_path_new,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,current_trial)

                            if fitness_value == 1:
                                print("Optimal solution found!")
                                break

                            elif(fitness_value > orig_fitness):
                                
                                #new_fitness =fitness_value
                                #orig_fitness = fitness_value
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                                print("reacheeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed")
                                #main_prompt ="\n Feedback for iteration "+str(current_trial)+": The changes you have made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time. Here are the code changes you made before:\n" + second_part
                                main_prompt = f"""
                                    Feedback for iteration {current_trial}: The changes you made have improved the correctness of the fix, but it is still not fully correct. 
                                    Please make only one change at a time to refine the solution further. Use the extra information below, along with previous details, to identify and correct the bug effectively.

                                    {prompt_final}

                                    Here are the code changes you in the previous iteration:
                                    {second_part}
                                    """
                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference. I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ", The changes you've made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time. Below are the code changes you previously suggested: \n "+second_part
                                #orig_fitness = new_fitness
                                #main_prompt = next_prompt
                                #context = [ {'role':'system', 'content':"You are a helpful assistant for fixing Verilog and system Verilog code."} ] 
                                print("context=")
                                print(context)
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                                current_trial += 1

                            
                            else:
                                

                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ". Below are the code changes you previously suggested: \n "+second_part
                                #main_prompt = "\n Feedback for iteration "+str(current_trial)+": The previous code changes you made did not resolve the functional issue and have worsened the code. Please review the bug and apply other changes that directly address the problem. Here are the code changes you made before:\n" + second_part
                                main_prompt = f"""
                                    Feedback for iteration {current_trial}: The previous code changes you made did not resolve the functional issue and have worsened the code. Please review the bug carefully and apply only changes that directly address the problem. Use the simulated output differences provided below, along with previous information, to detect and fix the bug effectively.

                                    {prompt_final}

                                    Here are the code changes you in the previous iteration:
                                    {second_part}
                                    """

                                #main_prompt =context+"\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at 0. try to Fix the code again "
                                #main_prompt =context
                                
                                print("Fitness value is not 1. Retrying...")
                                current_trial += 1
                        

                        elif(current_trial==2):
                        
                            prompt_final =all_input_feedback(file_name,buggy_src_file_path,directory_path_new,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,current_trial)

                            if fitness_value == 1:
                                print("Optimal solution found!")
                                break

                            elif(fitness_value > orig_fitness):
                                
                                #new_fitness =fitness_value
                                #orig_fitness = fitness_value
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                                print("reacheeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed")
                                #main_prompt ="\n Feedback for iteration "+str(current_trial)+": The changes you have made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time. Here are the code changes you made before:\n" + second_part
                                main_prompt = f"""
                                    Feedback for iteration {current_trial}: The changes you made have improved the correctness of the fix, but it is still not fully correct. Please make only one change at a time to refine the solution further. Use the extra information below, along with previous information, to identify and correct the bug effectively.

                                    The extra information includes:
                                    {prompt_final}

                                    Here are the code changes you made in the previous iteration:
                                    {second_part}
                                    """
                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference. I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ", The changes you've made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time. Below are the code changes you previously suggested: \n "+second_part
                                #orig_fitness = new_fitness
                                #main_prompt = next_prompt
                                #context = [ {'role':'system', 'content':"You are a helpful assistant for fixing Verilog and system Verilog code."} ] 
                                print("context=")
                                print(context)
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                                
                                current_trial += 1
                                


                            
                            else:
                                

                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ". Below are the code changes you previously suggested: \n "+second_part
                                main_prompt = f"""
                                    Feedback for iteration {current_trial}: The previous code changes you made did not resolve the functional issue and have worsened the code. Please review the bug carefully and apply only changes that directly address the problem. If you made a change to a line in the previous iteration and it didn't improve the code, do not repeat it in subsequent iterations. Use the simulated output differences and fault localization results provided below, along with previous information, to detect and fix the bug effectively.

                                    {prompt_final}

                                    Here are the code changes you made in the previous iteration:
                                    {second_part}
                                    """

                                #main_prompt =context+"\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at 0. try to Fix the code again "
                                #main_prompt =context
                                
                                print("Fitness value is not 1. Retrying...")
                                
                                current_trial += 1

                        elif(current_trial==3):
                        
                            prompt_final =all_input_feedback(file_name,buggy_src_file_path,directory_path_new,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,current_trial)

                            if fitness_value == 1:
                                print("Optimal solution found!")
                                break

                                                            
                            else:
                                

                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ". Below are the code changes you previously suggested: \n "+second_part
                                
                                main_prompt = f"""Feedback for iteration {current_trial}: This is your final iteration. All previous changes have not resolved the bug, so it is essential to think creatively and apply a fresh approach. Analyze your past responses carefully and consider the logic of the code along with the input information. The bug may lie in the lines identified through fault localization, so focus on making the correct change based on this insight.

                                            Avoid repeating changes that did not work and ensure that the fix fully addresses the issue. Provide the corrected Verilog code without extra words, comments, or explanations.

                                            """



                                #main_prompt =context+"\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at 0. try to Fix the code again "
                                #main_prompt =context
                                
                                print("Fitness value is not 1. Retrying...")
                                
                                current_trial += 1

                        elif(current_trial==4):
                            current_trial += 1
                            
                                
                                

                        print("current_trial = ")
                        print(current_trial)

                    
                    if current_trial == max_trials:
                        print("Maximum trials reached. Exiting loop.")
                    
                    
                    print("/////////////////////////////////////////////////////////")


                if (args.feedback_logic==2):
                    for iteration in range(1, iterations + 1):

                        max_trials = 5
                        current_trial = 0

                        while current_trial < max_trials:

                            if(current_trial==0):
                                iteration = iteration - 1
                                print(context)
                                directory_path=feedback_path_creation(file_name,directory_path,iteration)
                                main_prompt,_= feedback_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,iteration,args.feedback_logic,current_trial)
                            #else:
                            main_response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=collect_messages2(main_prompt)
                            
                            
                            #output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                            # azabat el input we el output fe el two functions dol
                            #replace iteration with current trials to update the files and have different files
                            output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing_feedback(main_response,current_trial,args.scenario_ID,file_name,model_type,directory_path,main_prompt)

                            #replace iteration with current trials to update the files and have different files
                            fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, current_trial,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                            
                            #for getting baseline fitness
                            #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                            ###########
                            #replace iteration with current trials to update the files and have different files
                            #save_output_from_gpt(file_name,args.scenario_ID,current_trial,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                            
                            print("/////////////////////////////////////////////////////////")
                            
                            if fitness_value == 1:
                                print("Optimal solution found!")
                                break
                            else:
                                #main_prompt = main_prompt+"\n I have send to you this code before and I wanted to reach value after simulation =1, but it is still not correct and these are the code changes you made before \n "+second_part
                                #main_prompt = main_prompt+"\n I previously sent you this code with the intention of achieving a simulation value of 1. However, despite the changes you made to the code, the simulation value is currently "+ str(fitness_value) + " these are the code changes you gave me before :\n "+second_part
                                #context[1]['content'] += "\n I previously sent you this code with the intention of achieving a simulation value of 1. However, despite the changes you made to the code, the simulation value is currently "+ str(fitness_value) + "  I want to receive different code changes as these changes did not yield the desired outcome. these are the code changes you gave me before :\n "+second_part
                               
                                main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + "and the response was : "+ main_response
                            

                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ". Below are the code changes you previously suggested: \n "+second_part
                                #main_prompt =context+"\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at 0. try to Fix the code again "
                                #main_prompt =context

                                print("Fitness value is not 1. Retrying...")
                                current_trial += 1

                        if current_trial == max_trials:
                            print("Maximum trials reached. Exiting loop.")
                        
                        fffff
                        print("/////////////////////////////////////////////////////////")

                if (args.feedback_logic==3): # logic number 3 where we add the prompt to the same user message
                    for iteration in range(1, iterations + 1):

                        max_trials = 5
                        current_trial = 0
                        while current_trial < max_trials:

                            if(current_trial==0):
                                iteration = iteration - 1
                                print(context)
                                directory_path=feedback_path_creation(file_name,directory_path,iteration)
                                main_prompt,orig_fitness = feedback_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,iteration,args.feedback_logic,current_trial)
                                context.append({'role':'user', 'content':f"{main_prompt}"})
                            #else:
                            main_response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=collect_messages3(main_prompt)
                            
                            
                            #output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                            # azabat el input we el output fe el two functions dol
                            #replace iteration with current trials to update the files and have different files
                            output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing_feedback(main_response,current_trial,args.scenario_ID,file_name,model_type,directory_path,main_prompt)

                            #replace iteration with current trials to update the files and have different files
                            fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, current_trial,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                            
                            #for getting baseline fitness
                            #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                            ###########
                            #replace iteration with current trials to update the files and have different files
                            save_output_from_gpt(file_name,args.scenario_ID,current_trial,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                            save_output_from_gpt_feedback(file_name, args.scenario_ID,(current_trial+1), number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='feedback_output_test3.csv')
                           
                            print("/////////////////////////////////////////////////////////")
                            
                            if fitness_value == 1:
                                print("Optimal solution found!")
                                break

                            elif(fitness_value > orig_fitness):
                                current_trial += 1
                                new_fitness =fitness_value
                                orig_fitness = fitness_value
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                                print("reacheeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed")
                                context[1]['content'] +="\n The changes you've made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time."
                               
                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference. I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ", The changes you've made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time. Below are the code changes you previously suggested: \n "+second_part
                                #orig_fitness = new_fitness
                                #main_prompt = next_prompt
                                #context = [ {'role':'system', 'content':"You are a helpful assistant for fixing Verilog and system Verilog code."} ] 
                                print("context=")
                                print(context)
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")

                            elif(fitness_value == new_fitness):
                                current_trial += 1
                                context[1]['content'] +="\n The changes you've made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time."
                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference. I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ", The changes you've made have improved the fix correctness, but it is still not fully correct. Please make only one change at a time. Below are the code changes you previously suggested: \n "+second_part
                                
                            else:
                                #main_prompt = main_prompt+"\n I have send to you this code before and I wanted to reach value after simulation =1, but it is still not correct and these are the code changes you made before \n "+second_part
                                #main_prompt = main_prompt+"\n I previously sent you this code with the intention of achieving a simulation value of 1. However, despite the changes you made to the code, the simulation value is currently "+ str(fitness_value) + " these are the code changes you gave me before :\n "+second_part
                                context[1]['content'] += "\n I previously sent you this code with the intention of achieving a simulation value of 1. However, despite the changes you made to the code, the simulation value is currently "+ str(fitness_value) + "  I want to receive different code changes as these changes did not yield the desired outcome. these are the code changes you gave me before :\n "+second_part
                                print("context_new :")
                                print(context)
                                #context.append({'role':'assistant', 'content':f"{second_part}"})
                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + "and the response was : "+ main_response
                            

                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ". Below are the code changes you previously suggested: \n "+second_part
                                #main_prompt =context+"\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at 0. try to Fix the code again "
                                #main_prompt =context

                                print("Fitness value is not 1. Retrying...")
                                current_trial += 1

                        if current_trial == max_trials:
                            print("Maximum trials reached. Exiting loop.")
                        
                        fffff
                        print("/////////////////////////////////////////////////////////")    

                if (args.feedback_logic==4):
                    for iteration in range(1, iterations + 1):

                        max_trials = 5
                        current_trial = 0
                        path_select =1

                        while current_trial < max_trials:

                            if(current_trial==0):
                                iteration = iteration - 1
                                print(context)
                                directory_path=feedback_path_creation(file_name,directory_path,iteration)
                                main_prompt,orig_fitness = feedback_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,iteration,args.feedback_logic,current_trial)
                                print("original buggy fitness=")
                                print(orig_fitness)
                                print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                                
                                
                            #else:
                        
                            main_response , second_part ,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=collect_messages4(main_prompt,path_select)
                            
                                
                            

                            # azabat el input we el output fe el two functions dol
                            #replace iteration with current trials to update the files and have different files
                            output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing_feedback(main_response,current_trial,args.scenario_ID,file_name,model_type,directory_path,main_prompt)
                            
                            #replace iteration with current trials to update the files and have different files
                            fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, current_trial,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                            
                            #for getting baseline fitness
                            #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                            ###########
                            #replace iteration with current trials to update the files and have different files
                            save_output_from_gpt(file_name,args.scenario_ID,current_trial,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                            save_output_from_gpt_feedback(file_name, args.scenario_ID,(current_trial+1), number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='feedback_output_test4.csv')
                            
                            print("/////////////////////////////////////////////////////////")
                            #fitness_value =-1
                            if fitness_value == 1:
                                print("Optimal solution found!")
                                break
                            elif(fitness_value > orig_fitness):
                                path_select = 1
                                current_trial += 1
                                next_prompt,new_fitness = feedback_scenario(output_file_name_after,output_file_path_after,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description,iteration,args.feedback_logic,current_trial)
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                                print("reacheeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed")
                                print("next prompt =")
                                print(next_prompt)
                                print("new_fitness =")
                                print(new_fitness)
                            
                                orig_fitness = new_fitness
                                main_prompt = next_prompt
                                context = [ {'role':'system', 'content':"You are a helpful assistant for fixing Verilog and system Verilog code."} ] 
                                print("context=")
                                print(context)
                                print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
                                
                            
                            else:

                                path_select = 0
                                #main_prompt = main_prompt+"\n I have send to you this code before and I wanted to reach value after simulation =1, but it is still not correct and these are the code changes you made before \n "+second_part
                                #main_prompt = main_prompt+"\n I previously sent you this code with the intention of achieving a simulation value of 1. However, despite the changes you made to the code, the simulation value is currently "+ str(fitness_value) + " these are the code changes you gave me before :\n "+second_part
                                #context[1]['content'] += "\n I previously sent you this code with the intention of achieving a simulation value of 1. However, despite the changes you made to the code, the simulation value is currently "+ str(fitness_value) + "  I want to receive different code changes as these changes did not yield the desired outcome. these are the code changes you gave me before :\n "+second_part
                                context[1]['content'] +="I previously shared this code snippet with the intention of achieving a successful simulation with a value of 1. However, despite implementing the changes you suggested, the simulation value remains at "+ str(fitness_value) + ". To clarify, simulation in this context refers to running a simulation of the code and obtaining the output, which is then compared with the expected output. Since the changes made didn't result in the desired outcome, I would appreciate alternative suggestions. Here are the code modifications you provided earlier for reference:\n\n"+second_part
                                print("context_new :")
                                print(context)
                                
                                #context.append({'role':'assistant', 'content':f"{second_part}"})
                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + "and the response was : "+ main_response
                            

                                #main_prompt ="\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at "+ str(fitness_value) + ". Below are the code changes you previously suggested: \n "+second_part
                                #main_prompt =context+"\n After generating the code, append a distinct section at the end, labeled Code Changes. In this section, outline and explain the variances between the corrected and original buggy code. Specify the exact changes made to the code, including line numbers. Ensure this section remains commented to avoid compiler interference.I previously shared this code with you aiming to achieve a simulation value of 1. However, despite the modifications you provided, the simulation value currently stands at 0. try to Fix the code again "
                                #main_prompt =context

                                print("Fitness value is not 1. Retrying...")
                                current_trial += 1

                        if current_trial == max_trials:
                            print("Maximum trials reached. Exiting loop.")
                        
                        fffff
                        print("/////////////////////////////////////////////////////////")

        
                   
        else:
            print("wrong value")

    elif (args.scenario_ID == 8):#8= Matching the bug description with the bug file 
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #future work
        elif(args.choose_file == "all") :
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = bug_description_matching(file_name,buggy_src_file_path,directory_path,bug_description)

                for iteration in range(1, iterations + 1):
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # how to fix the big codes with gpt4o
                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                        
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)

                    print("/////////////////////////////////////////////////////////")

        else:
            print("wrong value")


    elif (args.scenario_ID == 9):# 9= Changing the instrumnetal testbench 

        testbench_path_prev= None

        if(args.choose_file != "all"):
            file_name=args.choose_file
            #future work
        elif(args.choose_file == "all") :
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                if(test_bench_path != testbench_path_prev):

                    testbench_path_prev = test_bench_path

                    print("testbench_path_prev= ")
                    print(testbench_path_prev)
                    print("####################################################################")

                    iterations = args.number_iterations  # Set the number of iterations
                    prompt = testbench_changing(file_name,test_bench_path,directory_path)

                    for iteration in range(1, iterations + 1):
                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)


                        # how to fix the big codes with gpt4o
                        if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                            output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                        techniques_main_output(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)

                        try:
                            parsed_output = output_postprocess.split("start_code")[1].split("end_code")[0]
                            print("parsed_output=")
                            print(parsed_output)
                            output_postprocess= parsed_output
                            #parsed1 = json.loads(parsed_output)
                            #code_fixed = parsed1['fixed_code']
                            #code_fixed = parsed_output
                        except Exception as e:
                            print(f"An error occurred: {e}")

                        #parsed_output= output_postprocess.split("start_code")[1].split("end_code")[0]
                        #print("parsed_output=")
                        #print(parsed_output)
                        #parsed1=json.loads(parsed_output)
                        #code_fixed=parsed1['fixed_code']
                        #code_fixed=parsed_output

                        
                        print("output_postprocess=")
                        print(output_postprocess)

                        # azabat el input we el output fe el two functions dol
                        output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                            
                        fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                        
                        save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)

                        print("/////////////////////////////////////////////////////////")

                else:
                    pass


        else:
            print("wrong value")



    elif (args.scenario_ID == 10):#10= Running the main simulation and get the output
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #future work
        elif(args.choose_file == "all") :
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                #prompt = bug_description_matching(file_name,buggy_src_file_path,directory_path,bug_description)

                for iteration in range(1, iterations + 1):
                    #output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)


                    # all I need to change is the file path
                    original_file_path = os.path.join(PROJ_DIR, orig_file_name)

                    # here in order to run the original file or buggy file just change the path 

                    with open(original_file_path, 'r') as conf_file:
                        file_code = conf_file.read()

                    #with open(buggy_src_file_path, 'r') as conf_file:
                        #file_code = conf_file.read()

                    
                    f_name=orig_file_name.split('.v')[0]
                    f_name_buggy=file_name.split('.v')[0]
							
                    directory_preprocessing= directory_path +f_name_buggy+"/output_preprocessing"
                    directory_postprocessing= directory_path +f_name_buggy+"/output_postprocessing"
                    
                     
                
                    if not os.path.exists(directory_preprocessing):
                        os.makedirs(directory_preprocessing)
                        print(f"Directory '{directory_preprocessing}' created.")
                    else:
                        print(f"Directory '{directory_preprocessing}' already exists.")
                        
                        
                    if not os.path.exists(directory_postprocessing):
                        os.makedirs(directory_postprocessing)
                        print(f"Directory '{directory_postprocessing}' created.")
                    else:
                        print(f"Directory '{directory_postprocessing}' already exists.")
                        
                        
                        
                    init_output_file_name_before = f_name+'_iter_'+str(iteration)+'_Sc_ID_'+str(args.scenario_ID)+'.v'
                    init_output_file_name_after = f_name+'_iter_'+str(iteration)+'_Sc_ID_'+str(args.scenario_ID)+'.v'

                    output_file_path_before = os.path.join(directory_preprocessing, init_output_file_name_before) 
                    output_file_path_after = os.path.join(directory_postprocessing, init_output_file_name_after)
                    
                    
                   
                    
                    with open(output_file_path_after, 'w') as file_f:
                        file_f.write(str(file_code))
                        
                    #return init_output_file_name_after, output_file_path_after ,directory_postprocessing 




                    # I added this to check if the simulation is working correctly as I added the needed files and copied the correct code to the output from chatgpt file to make sure that I will get fitness =1
                    #output_file_name_after = "decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing/decoder_3_to_8_wadden_buggy1_iter_1_Sc_ID_0_model_type_gpt-3.5-turbo-1106.v"
                    #output_file_path_after_directory = "/scratch/abdelrahman/Cirfix/Automatic_Repair_LLM/Experimental_output/exp_num_exp3/SC_ID_0/decoder_3_to_8_wadden_buggy1/output_postprocessing"
                    
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,init_output_file_name_after, output_file_path_after,directory_postprocessing)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,0,0,0,0,0,0,0,fitness_value,simulation_status,simulation_time)
                    
                    print("/////////////////////////////////////////////////////////")
                    
                    
        else:
            print("wrong value")



    elif (args.scenario_ID == 11):#11= This experiment is to test the new test bench ---> so we would send buggy code + differences between oracle and simulation output (has more info)
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #future work
        elif(args.choose_file == "all") :
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = comp_tb_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # how to fix the big codes with gpt4o
                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                         
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)

                    #save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_unstructured_4o.csv')
                    save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_structured_4o.csv')
                    

                    print("/////////////////////////////////////////////////////////")
                    
        else:
            print("wrong value")

    elif (args.scenario_ID == 12):#12= This experiment same as number 11 but adding the lines of code that the bug might originate from using fault localization
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #future work
        elif(args.choose_file == "all") :
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = detecting_lines_scenario(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # how to fix the big codes with gpt4o
                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                         
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    #save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_unstructured_4o.csv')
                    save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_structured_4o.csv')
                    
                    print("/////////////////////////////////////////////////////////")
                    
                    
        else:
            print("wrong value")

    elif (args.scenario_ID == 13):#13= This experiment same as number 12 but adding the lines of code that the bug might originate from using fault localization
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #future work
        elif(args.choose_file == "all") :
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = lines_with_Fault(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # how to fix the big codes with gpt4o
                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                         
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                    
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    #save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_unstructured_4o.csv')
                    save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_structured_4o.csv')
                    
                    print("/////////////////////////////////////////////////////////")
                    
                    
        else:
            print("wrong value")


    elif (args.scenario_ID == 14):#14= This experiment add all the input information 
        if(args.choose_file != "all"):
            file_name=args.choose_file
            #future work
        elif(args.choose_file == "all") :
            for index, row in islice(df.iterrows(), start_index, None):   
                file_name = row['file_name']  # Replace 'file_column' with the actual column name containing file names
                buggy_src_file_path = row['buggy_src_file']
                EVAL_SCRIPT = row['eval_script']
                test_bench_path= row['test_bench']
                orig_file_name=row['orig_file']
                PROJ_DIR= row['proj_dir']
                oracle_path=row['oracle']
                bug_description=row['simple_bug_description']
                print("bug_description = ",bug_description)
                print("buggy_src_file_path = ",buggy_src_file_path)
                print("EVAL_SCRIPT = ",EVAL_SCRIPT)
                print("test_bench_path = ",test_bench_path)
                print("orig_file_name = ",orig_file_name)
                print("PROJ_DIR = ",PROJ_DIR)
                print("oracle_path = ",oracle_path)
                
                iterations = args.number_iterations  # Set the number of iterations
                prompt = all_input_information(file_name,buggy_src_file_path,directory_path,test_bench_path,oracle_path,EVAL_SCRIPT, orig_file_name, PROJ_DIR,bug_description)
               
                for iteration in range(1, iterations + 1):
                    
                    output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=send_prompt_chatgpt(prompt)

                    # how to fix the big codes with gpt4o
                    if(number_tokens_output>=4096 and model_type== "gpt-4o-2024-05-13"):

                        output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time=small_feedback_big_file(output_postprocess,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,prompt)


                    # azabat el input we el output fe el two functions dol
                    output_file_name_after,output_file_path_after,output_file_path_after_directory=gpt_output_postprocessing(output_postprocess,iteration,args.scenario_ID,file_name,model_type,directory_path)
                         
                    fitness_value,simulation_status,simulation_time= run_simulation(EVAL_SCRIPT, orig_file_name, iteration,test_bench_path, oracle_path,PROJ_DIR,output_file_name_after, output_file_path_after,output_file_path_after_directory)
                    
                    #for getting baseline fitness
                    #fitness_value,simulation_status,simulation_time=run_baseline_simulation(EVAL_SCRIPT, orig_file_name,test_bench_path, oracle_path,PROJ_DIR,buggy_src_file_path,file_name)

                    ###########
                   
                    save_output_from_gpt(file_name,args.scenario_ID,iteration,number_tokens_input,number_tokens_output,cost_input,cost_output,cost_total,model_type,time,fitness_value,simulation_status,simulation_time)
                    #save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_unstructured_4o.csv')
                    save_output_from_gpt2(file_name, args.scenario_ID, iteration, number_tokens_input, number_tokens_output, cost_input, cost_output, cost_total, model_type, time, fitness_value, simulation_status, simulation_time, output_file='few_structured_4o.csv')
                    
                    print("/////////////////////////////////////////////////////////")
                    
                    
        else:
            print("wrong value")


    # Your code here
    print("Arguments:", args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your script')
    
    # Define command-line arguments
    parser.add_argument('pandas_csv_path', type=str, help='Path to the csv file')
    parser.add_argument('number_iterations', type=int, help='Number of iterations to repeat passing the same prompt to gpt')
    parser.add_argument('choose_file', type=str, help='choose file name to test')#add specific file name or "all" to process on all files
    parser.add_argument('scenario_ID', type=int, help='chooses the prompt scneario')
    parser.add_argument('experiment_number', type=str, help='write the experiment number')
    parser.add_argument('feedback_logic', type=int, help='choose your feedback logic')
   # parser.add_argument('--optional_arg', type=float, default=1.0, help='Description of optional_arg')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
