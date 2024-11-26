#!/usr/bin/env python

import sys
import json
from pair import pair

def main(uis):
    if "pair_example" in uis:
        with open("config.json", "r") as fh:
            data_params = json.load(fh)
        target_prompt = pair(**data_params)
        print(target_prompt)

if __name__ == '__main__':
    user_inputs = sys.argv[1:]
    main(user_inputs)