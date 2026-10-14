"""Static and import compatibility checks; never executes model training scripts.

Intentionally imports only the modules/classes referenced by top-level import
statements, rather than importing scripts that start training at module import.
"""
import ast
import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_2_advanced_models"))
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

TARGETS = [
    ROOT / "scripts_1_benchmark _models",
    ROOT / "scripts_2_advanced_models",
]
ALLOWED_KEYWORDS = {
    "Seq2SeqTrainingArguments": set(inspect.signature(Seq2SeqTrainingArguments).parameters),
    "Seq2SeqTrainer": set(inspect.signature(Seq2SeqTrainer).parameters),
}
checked_files = 0
checked_imports = set()
problems = []

for directory in TARGETS:
    for script in sorted(directory.rglob("*.py")):
        checked_files += 1
        try:
            tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        except Exception as exc:
            problems.append(f"{script.relative_to(ROOT)}: parse failure: {exc}")
            continue

        # Only statements at module level. Importing scripts directly would
        # initiate dataset downloads and training, so do NOT do that here.
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    checked_imports.add((alias.name, None))
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    if alias.name != "*":
                        checked_imports.add((node.module, alias.name))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            allowed = ALLOWED_KEYWORDS.get(node.func.id)
            if allowed is None:
                continue
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in allowed:
                    problems.append(
                        f"{script.relative_to(ROOT)}:{kw.value.lineno}: "
                        f"{node.func.id} does not accept {kw.arg!r}"
                    )

for module_name, symbol in sorted(checked_imports):
    try:
        module = importlib.import_module(module_name)
        if symbol is not None:
            getattr(module, symbol)
    except Exception as exc:
        target = f"{module_name}.{symbol}" if symbol else module_name
        problems.append(f"Import {target}: {type(exc).__name__}: {exc}")

if problems:
    print("FAILED import/API checks:")
    for problem in problems:
        print(" -", problem)
    sys.exit(1)
print(f"PASS: {checked_files} Python files inspected, "
      f"{len(checked_imports)} unique imports checked, and "
      "Seq2SeqTrainer/TrainingArguments keyword names validated.")
