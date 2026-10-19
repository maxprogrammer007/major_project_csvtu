from pathlib import Path

target_file = Path("writeup_2026-09-16/four_paradigm_master_table.md")
lines = target_file.read_text(encoding="utf-8").splitlines()

# Find the start of Table 3
start_idx = -1
for i, l in enumerate(lines):
    if l.startswith("| Condition | Token Source"):
        start_idx = i
        break

if start_idx != -1:
    end_idx = start_idx
    while end_idx < len(lines) and lines[end_idx].startswith("|"):
        end_idx += 1
    
    new_table_lines = [
        "| Condition | Token Representation | Total TPs | Micro Recall | Macro Recall | Precision | F1 | TEDS-S (Primary) | GriTS-Top (Primary) | GriTS-Con (Within-Token) | Solve Time |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        "| **Raw** | None | 1,833 | 32.47% | 33.81% | 27.16% | 28.42% | 0.6680 | 0.7575 | -- | -- |",
        "| **H1-GT** | Oracle GT Tokens | 1,203 | 21.31% | 23.06% | 27.38% | 24.11% | 0.6888 | 0.7633 | 0.6042 | -- |",
        "| **H1-Word-Boxes** | Geometric Word-Boxes | 1,336 | 23.66% | 26.02% | 27.89% | 26.29% | **0.7318** | **0.7908** | -- (omit dummy) | -- |",
        "| **H2-GT** | Oracle GT Tokens | 1,839 | 32.57% | 34.02% | 27.72% | 28.87% | 0.6733 | 0.7594 | **0.6212** | 61.6 ms |",
        "| **H2-Word-Boxes** | Geometric Word-Boxes | **1,836** | **32.52%** | **33.96%** | 27.49% | **28.72%** | 0.6725 | 0.7605 | -- (omit dummy) | 60.3 ms |",
        "| **Greedy (Unnormalized)**| Geometric Word-Boxes | 504 | 8.93% | 10.68% | 13.15% | 10.74% | 0.4230 | 0.5629 | -- (omit dummy) | 0.8 ms |",
        "| **Greedy (Normalized)**  | Geometric Word-Boxes | 1,835 | 32.50% | 33.94% | 27.48% | 28.71% | 0.6725 | 0.7605 | -- (omit dummy) | 0.9 ms |"
    ]
    lines = lines[:start_idx] + new_table_lines + lines[end_idx:]
    print(f"Replaced Table 3 from line {start_idx} to {end_idx}")

# Also update the note below Table 3
text = "\n".join(lines)
text = text.replace(
    "*Note on GriTS-Con for OCR conditions:*",
    "*Note on Metric Architecture & Word-Boxes:*"
)
text = text.replace(
    "GriTS-Con is omitted for OCR conditions",
    "GriTS-Con is strictly omitted for geometric word-box conditions"
)
text = text.replace("$H2_{\\text{OCR}}$", "$H2_{\\text{Word-Boxes}}$")
text = text.replace("$H1_{\\text{OCR}}$", "$H1_{\\text{Word-Boxes}}$")

target_file.write_text(text, encoding="utf-8")
print("Saved updated file.")
