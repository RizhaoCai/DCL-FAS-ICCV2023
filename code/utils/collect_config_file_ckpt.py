import pandas as pd
import re
import os
from glob import glob

import argparse


def get_csv_list(pattern):
    # for root, dirs, files in os.walk('experiments'):
    #     for files
    # re_template = pass

    # all_csv_list = glob('output/*/*/test/*.csv')
    # #pattern = re.compile(keyword)
    # csv_list = filter(lambda x:bool(re.match(pattern,x)),all_csv_list)
    # csv_list = list(csv_list)

    csv_list = []
    for root, dirs, files in os.walk('output'):
        for f in files:
            f_name = os.path.join(root, f)
            if 'csv' in f_name and bool(re.search(pattern,f_name)):
                csv_list.append(f_name)
    return csv_list


def combine_csv(output_path, csv_list):
    with open(output_path,'w') as new_csv:
        for csv_path in csv_list:
            print('Read contents csv:',csv_path)
            with open(csv_path,'r') as old_csv:
                contents = csv_path.replace('output/vit_convpass/vit_base_patch16_224_Adam_pretrain_update_convpass/','')+old_csv.read()
                # import pdb; pdb.set_trace()
                new_csv.write(contents)
    print('Write contents to new csv:', output_path)

def main():
    parser = argparse.ArgumentParser(description='Args for collecting csv results')
    parser.add_argument('-o', type=str, help='Output files path', dest='output')
    parser.add_argument('-p', type=str,help='Regex patterns', dest='pattern')
    args = parser.parse_args()

    output_names = args.output if args.output \
        else 'csv_results/BASE_SIW+HiFi+CASIA.csv'
    # ResNet-P4
    pattern =  args.pattern if args.pattern \
            else r'(.+)convpass/(.+)test_frame_metrics.csv'
    #pattern =  r'(.+)resnet18/OULU\-TRAIN\-P4\-(.+)'

    csv_list = get_csv_list(pattern)
    assert csv_list, "NULL csv_list"
    print(csv_list)
    combine_csv(output_names, csv_list)

if __name__ == '__main__':
    main()