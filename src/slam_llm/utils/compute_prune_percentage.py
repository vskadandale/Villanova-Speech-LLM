import re
import sys
import os

def calculate_pruning_stats(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Regex to find patterns like Pruned_Frames:[ 1 ]/[ 271 ]
    # Handles potential spaces inside brackets
    pattern = r"Pruned_Frames:\[\s*(\d+)\s*\]/\[\s*(\d+)\s*\]"
    
    pruned_list = []
    total_list = []
    ratios = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    p, t = int(match.group(1)), int(match.group(2))
                    if t > 0:
                        pruned_list.append(p)
                        total_list.append(t)
                        ratios.append(p / t)

        if not pruned_list:
            print("No pruning data found in the provided log file.")
            return

        # Macro Average: Average of the ratios per step
        macro_avg = sum(ratios) / len(ratios)

        # Micro Average: Sum of all pruned / Sum of all totals
        micro_avg = sum(pruned_list) / sum(total_list)

        print("-" * 30)
        print(f"Analysis for: {file_path}")
        print(f"Total Steps Processed: {len(pruned_list)}")
        print(f"Total Pruned Frames:  {sum(pruned_list)}")
        print(f"Total Frames Scanned: {sum(total_list)}")
        print("-" * 30)
        print(f"Macro Average: {macro_avg:.6%}")
        print(f"Micro Average: {micro_avg:.6%}")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <path_to_log_file>")
    else:
        calculate_pruning_stats(sys.argv[1])
