import argparse
import csv
import json
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

random.seed(42)


def convert_dialogue_to_csv(path, dtype):
    p = Path(os.path.join(path, dtype))
    print(p)
    dialogue_summary = []
    for x in p.glob('*.json'):
        print(x)
        with open(x) as f:
            data = json.load(f)
            dialogues = data.get('data', [])
    
            for d in dialogues:
                body = d.get('body', {})
                utterances, tmp, previous_id = [], [], None
                for turn in body.get('dialogue', []):
                    participantID = turn.get('participantID', '')
                    utterance = turn.get('utterance').replace('\n', ' ').strip()
                    if previous_id != participantID:
                        if tmp:
                            utterances.append(f'{previous_id}: {" ".join(tmp)}')
                        tmp = []
                    tmp.append(utterance) 
                    previous_id = participantID
    
                dialogue = '<sep>'.join(utterances)
                summary = body.get('summary', '').replace('\n', ' ').strip()
                if summary:
                    dialogue_summary.append((dialogue, summary))

    random.shuffle(dialogue_summary)
    return dialogue_summary


def write_csv(data, path, dtype):
    with open(os.path.join(path, f'{dtype}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "dialogue", "summary"])
        for i, d in enumerate(data, 1):
            writer.writerow([str(i)] + list(d))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--output")
    args = parser.parse_args()

    data_for_post_train = convert_dialogue_to_csv(args.path, "train")
    data_for_finetune_and_eval = convert_dialogue_to_csv(args.path, "valid")

    with open(os.path.join(args.output, 'dialogue.json'), 'w') as f:
        for dialogue, _ in data_for_post_train:
            json.dump({"dialogue": dialogue}, f, ensure_ascii=False)
            f.write('\n')


    train, test = train_test_split(data_for_finetune_and_eval, test_size=0.1)
    train, valid = train_test_split(train, test_size=0.1)
    write_csv(train, args.output, 'train')
    write_csv(valid, args.output, 'valid')
    write_csv(test, args.output, 'test')


