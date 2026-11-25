"""Run the chosen prompt on the submitted test video sample."""

from pathlib import Path

from llm_harness import PROMPTS, SELECTED_PROMPT, predict


if __name__ == "__main__":
    sample = Path(__file__).resolve().parent / "01.mp4"
    prediction, _, _ = predict(sample, PROMPTS[SELECTED_PROMPT])
    print(f"{sample.name}: {prediction or 'INVALID'}")
