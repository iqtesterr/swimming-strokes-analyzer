import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ('backstroke', 'breaststroke', 'butterfly', 'freestyle')


def predictions(path, column):
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            example_id = row['example_id']
            result[example_id] = row[column]
    return result


def report(name, guesses, truth):
    # Overall correctness
    total = len(truth)
    correct = 0
    for key in truth:
        if guesses.get(key) == truth[key]:
            correct += 1

    print(f'{name}: {correct}/{total} = {correct/total:.2%}')

    # Strokes correctness
    for label in LABELS:
        hits = 0
        count = 0
        for key in truth:
            if truth[key] == label:
                count += 1
                if guesses.get(key) == label:
                    hits += 1
        print(f'  {label} recall: {hits}/{count} = {hits/count:.2%}')
    invalid = sum(guesses.get(key) not in LABELS for key in truth)
    print(f'  invalid (including unknown or missing): {invalid}\n')


def main():
    truth = predictions(ROOT/'ground_truth.csv', 'label')
    human = predictions(ROOT/'human_eval.csv', 'human_prediction')
    symbolic = predictions(ROOT/'symbolic_eval.csv', 'prediction')
    ml = predictions(ROOT/'ml_eval.csv', 'prediction')
    llm = predictions(ROOT/'llm_eval.csv', 'prediction')
    human_truth = {key: truth[key] for key in human}
    report('Human subset', human, human_truth)
    report('Symbolic on the same subset', symbolic, human_truth)
    report('Symbolic on the full test set', symbolic, truth)
    report('ML on the same subset', ml, human_truth)
    report('ML on the full test set', ml, truth)
    report('LLM on the same subset', llm, human_truth)
    report('LLM on the full test set', llm, truth)


if __name__ == '__main__':
    main()
