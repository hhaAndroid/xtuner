import argparse
import os
import csv
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')
    parser.add_argument('result_path', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    all_files = os.listdir(args.result_path)
    overall_values = {}
    for file in all_files:
        if file.endswith('.csv'):
            path = os.path.join(args.result_path, file)
            with open(path, mode='r', newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                headers = next(csvreader)
                if 'HallusionBench_score.csv' in path:
                    row = next(csvreader)
                    acc = (float(row[1]) + float(row[2]) + float(row[3])) / 3
                    overall_values[file] = round(acc, 1)
                elif 'MME_score.csv' in path:
                    row = next(csvreader)
                    perception = float(row[0])
                    reasoning = float(row[1])
                    overall = (perception + reasoning) / 2800
                    overall_values[file] = round(overall*100, 1)
                    print('mme', int(perception), int(reasoning))
                elif 'MMMU_DEV_VAL_acc.csv' in path:
                    for row in csvreader:
                        if row[0] == 'validation':
                            overall_values[file] = round(float(row[1])*100, 1)
                            break
                else:
                    overall_index = headers.index('Overall')
                    row = next(csvreader)
                    if 'TextVQA_VAL_acc.csv' in path or \
                            'DocVQA_VAL_acc.csv' in path or \
                            'InfoVQA_VAL_acc.csv' in path or \
                            'ChartQA_TEST_acc.csv' in path:
                        overall_values[file] = round(float(row[overall_index]), 1)
                    else:
                        overall_values[file] = round(float(row[overall_index])*100, 1)

    # json
    for file in all_files:
        if file.endswith('.json'):
            if 'OCRBench_score.json' in file:
                path = os.path.join(args.result_path, file)
                with open(path, mode='r') as f:
                    data = json.load(f)
                    overall_values[file] = round(data['Final Score Norm'], 1)

    print(overall_values)
