import subprocess

# Setting parameters

script_name = "Graph_enhance_NL2LTL.py"
n_paths = 5
n_trials = 3
nl_file = "SSCS.txt"
save_csv_file = "X_results.csv"
api = ["sk-X","sk-X","sk-X","sk-X","sk-X"]
model = ["gpt-X", "gpt-X", "claude-X", "deepseek-X","gemini-X"]
#
# api = ["sk-X"]
# model = ["gpt-X"]
#

# Read text files line by line and execute commands
with open(nl_file, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        text = line.strip()
        if not text:
            continue
        command = [
            "python", script_name,
            "--texts", text,
            "--save_csv_file", save_csv_file,
            "--n_paths", str(n_paths),
            "--n_trials", str(n_trials),
            "--model", str(model),
            "--api", str(api)
        ]
        # print(f"\n[Line {line_num}] Executing: {' '.join(command)}")
        # result = subprocess.run(command, capture_output=True, text=True)
        result = subprocess.run(command)

        # print("Output:\n", result.stdout)
        if result.stderr:
            print("Error:\n", result.stderr)
