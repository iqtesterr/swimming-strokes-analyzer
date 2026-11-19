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
    total = len(guesses)
    correct = 0
    for key in guesses:
        if guesses[key] == truth[key]:
            correct += 1

    print(f'{name}: {correct}/{total} = {correct/total:.1%}')

    # Strokes correctness
    for label in LABELS:
        hits = 0
        count = 0
        for key in guesses:
            if truth[key] == label:
                count += 1
                if guesses[key] == label:
                    hits += 1
        print(f'  {label}: {hits}/{count}')
    unknown = sum(label == 'unknown' for label in guesses.values())
    print(f'  unknown: {unknown}')


def main():
    truth = predictions(ROOT/'ground_truth.csv', 'label')
    human = predictions(ROOT/'human_eval.csv', 'human_prediction')
    symbolic = predictions(ROOT/'symbolic_eval.csv', 'prediction')
    report('Human subset', human, truth)
    report('Symbolic on the same subset', {key: symbolic[key] for key in human}, truth)
    report('Symbolic on the full test set', symbolic, truth)


if __name__ == '__main__':
    main()
