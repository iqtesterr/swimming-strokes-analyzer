import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import ollama

from video_loader import STROKES, split_frames

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data"
MODEL = "gemma3:12b"
OPTIONS = {
    "temperature": 0,
    "seed": 5863,
    "num_ctx": 65536,
}
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": list(STROKES)}},
    "required": ["answer"]
}
N_PER_STROKE = 20
SELECTED_PROMPT = "v6"

PROMPTS = {
    # baseline
    "v0": """<task>
Classify the swimming stroke shown in these 16 frames from a single three seconds video.
</task>

<note>
The attached images are in chronological order.
</note>

<label>
backstroke
breaststroke
butterfly
freestyle
</label>

<output>
Return JSON with one field named "answer", containing one label from above.
</output>""",

    # change: reverse the label order
    # hypothesis: if the model is not guessing,
    #             the result should stay unchanged
    "v1": """<task>
Classify the swimming stroke shown in these 16 frames from a single three seconds video.
</task>

<note>
The attached images are in chronological order.
</note>

<label>
freestyle
butterfly
breaststroke
backstroke
</label>

<output>
Return JSON with one field named "answer", containing one label from above.
</output>""",

    # change: add motion cues
    # hypothesis: this helps distinguish similar poses
    "v2": """<task>
Classify the swimming stroke shown in these 16 frames from a single three seconds video.
</task>

<note>
The attached images are in chronological order.
Use arm and leg movement across frames. 
Freestyle and backstroke have alternating arms. 
One has head facing downward, and one has head facing upward.
Breaststroke has a simultaneous arm pull and frog kick.
Butterfly has simultaneous arm movement and a dolphin kick.
</note>

<label>
backstroke
breaststroke
butterfly
freestyle
</label>

<output>
Return JSON with one field named "answer", containing one label from above.
</output>""",

    # change: add a distraction warning
    # hypothesis: reduce the background-based guesses
    "v3": """<task>
Classify the swimming stroke shown in these 16 frames from a single three seconds video.
</task>

<note>
The attached images are in chronological order.
Use arm and leg movement across frames. 
Freestyle and backstroke have alternating arms. 
One has head facing downward, and one has head facing upward.
Breaststroke has a simultaneous arm pull and frog kick.
Butterfly has simultaneous arm movement and a dolphin kick.
Please ignore lighting, lane lines, and clothing.
</note>

<label>
backstroke
breaststroke
butterfly
freestyle
</label>

<output>
Return JSON with one field named "answer", containing one label from above.
</output>""",

    # change: ask the model to think
    # hypothesis: hope this stop model from predicting the same stroke
    "v4": """<task>
Classify the swimming stroke shown in these 16 frames from a single three seconds video.
Think before you made a classification.
</task>

<note>
The attached images are in chronological order.
Use arm and leg movement across frames. 
Freestyle and backstroke have alternating arms. 
One has head facing downward, and one has head facing upward.
Breaststroke has a simultaneous arm pull and frog kick.
Butterfly has simultaneous arm movement and a dolphin kick.
Please ignore lighting, lane lines, and clothing.
</note>

<label>
backstroke
breaststroke
butterfly
freestyle
</label>

<output>
Return JSON with one field named "answer", containing one label from above.
</output>""",

    # change: improve the notes, emphasize the importance to analyze base on the chronological order
    # hypothesis: hope this stop model from predicting the same stroke
    "v5": """<task>
Classify the swimming stroke shown in these 16 frames from a single three seconds video.
Think before you made a classification.
</task>

<note>
The attached images are in chronological order.
Use arm and leg movement across frames, see all the images and analyze based on chronological order of the frames. 
Freestyle and backstroke have alternating arms. 
One has head facing downward, and one has head facing upward.
Breaststroke has a simultaneous arm pull and frog kick.
Butterfly has simultaneous arm movement and a dolphin kick.
Please ignore lighting, lane lines, and clothing.
Do not give prediction without seeing all the frames.
</note>

<label>
backstroke
breaststroke
butterfly
freestyle
</label>

<output>
Return JSON with one field named "answer", containing one label from above.
</output>""",
    # SCHEMA already defines the labels, so do not repeat them in the prompt.
    "v6": """<task>
Classify the swimming stroke shown in these 16 frames from a single three seconds video.
</task>

<note>
The attached images are in chronological order.
</note>

<output>
Return JSON with one field named "answer", containing only one label.
</output>""",
    "v7": """<task>
Classify the swimming stroke shown in these 16 frames from a single three seconds video.
</task>

<note>
The attached images are in chronological order.
Use arm and leg movement across frames, see all the images and analyze based on chronological order of the frames. 
Freestyle and backstroke have alternating arms. 
One has head facing downward, and one has head facing upward.
Breaststroke has a simultaneous arm pull and frog kick.
Butterfly has simultaneous arm movement and a dolphin kick.
Please ignore lighting, lane lines, and clothing.
Do not give prediction without seeing all the frames.
</note>

<output>
Return JSON with one field named "answer", containing only one label.
</output>""",
}


def converted_images(path):
    """Cache 16 chronological 640x360 frames, one folder per video."""
    relative = path.relative_to(DATA) if path.is_relative_to(DATA) else Path("sample")/path.name
    folder = DATA/"llm"/relative.with_suffix("")
    paths = [folder/f"{i:02d}.jpg" for i in range(1, 17)]
    if not all(p.exists() for p in paths):
        folder.mkdir(parents=True, exist_ok=True)
        for target, frame in zip(paths, split_frames(path)):
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(target), frame)
    return [str(p) for p in paths]


def examples(split):
    """Choose 20 videos per class for train/validation, or the full test set."""
    rows = []
    if split == "test":
        return [{"id": path.stem, "path": path}
                for path in sorted((DATA/"test").glob("*/*.mp4"))]
    for stroke in STROKES:
        paths = sorted((DATA/split/stroke).glob("*.mp4"))
        if split == "train":
            paths = random.Random(f"CSC211-wang5863-{stroke}").sample(paths, N_PER_STROKE)
        elif split == "validation":
            paths = random.Random(f"CSC211-wang5863-validation-{stroke}").sample(paths, N_PER_STROKE)
        rows.extend({"id": path.stem, "path": path, "label": stroke} for path in paths)
    return rows


def predict(path, prompt):
    """Return a stroke, or None for an invalid answer, plus the raw response."""
    image_paths = converted_images(path)
    start = time.perf_counter()
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt, "images": image_paths}],
        format=SCHEMA,
        options=OPTIONS,
        think=False,
    )
    seconds = time.perf_counter() - start
    try:
        answer = json.loads(response.message.content)["answer"]
    except (ValueError, KeyError, TypeError):
        answer = None
    return (answer if answer in STROKES else None), response, seconds


def run(version, rows, split):
    """
    Save raw responses in JSONL, with the first line records the setup.
    It also resumes a run without repeating calls.
    """
    prompt = PROMPTS[version]
    output = ROOT/"eval"/"llm"/MODEL.replace(":", "_")/f"{version}_{split}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    setup = {"model": MODEL, "options": OPTIONS, "schema": SCHEMA,
             "prompt": prompt, "conversion": "16 separate chronological frames, 640x360",
             "videos": [str(row["path"].relative_to(ROOT)) for row in rows]}
    saved = []
    if output.exists():
        with output.open(encoding="utf-8") as file:
            lines = [json.loads(line) for line in file if line.strip()]
        if lines[0] == setup:
            saved = lines[1:]

    print(f"\n{version}: exact prompt (image attached, filename excluded):\n{prompt}\n")
    with output.open("w", encoding="utf-8") as file:
        file.write(json.dumps(setup) + "\n")
        for result in saved:
            file.write(json.dumps(result) + "\n")
        for row in rows[len(saved):]:
            prediction, response, seconds = predict(row["path"], prompt)
            result = {"id": row["id"], "prediction": prediction,
                      "raw": response.message.content, "seconds": round(seconds, 2),
                      "prompt_tokens": response.prompt_eval_count}
            file.write(json.dumps(result) + "\n")
            file.flush()
            saved.append(result)
            print(f"{row['id']}: {prediction or 'INVALID'} ({seconds:.1f} s)", flush=True)
    return saved


def record_prompt(version, split, accuracy, invalid, recalls):
    """Append one prompt evaluation to the model's train.csv or val.csv."""
    folder = ROOT/"eval"/"llm"/MODEL.replace(":", "_")
    folder.mkdir(parents=True, exist_ok=True)
    output = folder/("train.csv" if split == "train" else "val.csv")
    row = {"prompt_version": version, "accuracy": accuracy, "invalid": invalid}
    for stroke in STROKES:
        row[f"{stroke}_recall"] = recalls[stroke]
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def compare(versions, split):
    rows = examples(split)
    outcomes = {}
    for version in versions:
        results = run(version, rows, split)
        correct = [result["prediction"] == row["label"]
                   for result, row in zip(results, rows)]
        outcomes[version] = correct
        invalid = sum(result["prediction"] is None for result in results)
        seconds = sum(result["seconds"] for result in results) / len(results)
        print(f"{version}: accuracy {sum(correct)}/{len(rows)} = {sum(correct)/len(rows):.1%}; "
              f"invalid {invalid}; {seconds:.1f} s/example")
        recalls = {}
        for stroke in STROKES:
            group = [ok for ok, row in zip(correct, rows) if row["label"] == stroke]
            recalls[stroke] = sum(group)/len(group)
            print(f"  {stroke} recall: {sum(group)}/{len(group)}")
        record_prompt(version, split, sum(correct)/len(rows), invalid, recalls)
    base = versions[0]
    for version in versions[1:]:
        fixes = sum(new and not old for old, new in zip(outcomes[base], outcomes[version]))
        breaks = sum(old and not new for old, new in zip(outcomes[base], outcomes[version]))
        print(f"{version} vs {base}: {fixes} fixes, {breaks} breaks")


def test_set(version):
    """Save the predictions of test_set to llm_eval.csv."""
    results = run(version, examples("test"), "test")
    output = ROOT/"llm_eval.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("example_id", "prediction"))
        for result in results:
            writer.writerow((result["id"], result["prediction"] or "INVALID"))
    print(f"Wrote {len(results)} predictions to {output}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        compare([SELECTED_PROMPT], "train")
    elif sys.argv[1] == "val":
        compare(sys.argv[2:], "validation")
    elif sys.argv[1] == "test":
        test_set(sys.argv[2])
