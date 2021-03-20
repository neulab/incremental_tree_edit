from edit_components.dataset import OrderedDict, json

import sys

assert len(sys.argv) == 4

wiki_insert_file = sys.argv[1]
wiki_del_file = sys.argv[2]
output_file = sys.argv[3]

with open(output_file, 'w') as f:
    for i, line in enumerate(open(wiki_insert_file, errors='ignore')):
        data = line.strip().split('\t')
        data = [x.lower() for x in data]

        entry = OrderedDict(Id=f'Ins_{i}',
                            PrevCodeChunkTokens=data[0].split(' '),
                            UpdatedCodeChunkTokens=data[2].split(' '),
                            PrevCodeChunk=data[0] + ' ||| ' + data[1],
                            UpdatedCodeChunk=data[2],
                            PrecedingContext=[],
                            SucceedingContext=[])

        f.write(json.dumps(entry) + '\n')

    for i, line in enumerate(open(wiki_del_file, errors='ignore')):
        data = line.strip().split('\t')
        data = [x.lower() for x in data]

        entry = OrderedDict(Id=f'Del_{i}',
                            PrevCodeChunkTokens=data[0].split(' '),
                            UpdatedCodeChunkTokens=data[2].split(' '),
                            PrevCodeChunk=data[0] + ' ||| ' + data[1],
                            UpdatedCodeChunk=data[2],
                            PrecedingContext=[],
                            SucceedingContext=[])

        f.write(json.dumps(entry) + '\n')
