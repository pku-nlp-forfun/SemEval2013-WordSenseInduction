# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-05-18 22:30:39
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-05-21 11:02:47
import os
import subprocess
import sys
import tempfile
from typing import Dict

import numpy as np


def evaluateFromFile(input_path: str, log_path: str = "log/evaluation.log"):
    ''' evaluation from file '''
    with open(input_path, 'r') as f:
        test = [ii.strip().split() for ii in f.readlines()]
    data = {ii[1]: {jj.split('/')[0]: float(jj.split('/')[1])
                    for jj in ii[2:]} for ii in test}

    handleScore(data, input_path, log_path)


def handleScore(data: Dict[str, Dict[str, float]], evaluationId: str, log_path: str = "log/evaluation.log", outPath: str = None) -> Dict[str, Dict[str, float]]:
    ''' handle score '''
    scores = evaluate_labeling(
        './SemEval-2013-Task-13-test-data', data, outPath)
    jaccard = scores['all']['jaccard-index']
    pos = scores['all']['pos-tau']
    WNDC = scores['all']['WNDC']
    fnmi = scores['all']['FNMI']
    fbc = scores['all']['FBC']
    msg = 'Result: jaccard|pos  |WNDC |FNMI|FBC  |AVG  |\n          '
    for ii in scores['all'].values():
        msg += '{:.2f}|'.format(ii * 100)
    msg += '{:.2f}|'.format(np.sqrt(fnmi * fbc) * 100)

    # mkdir if the folder don't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write('{}: {}\n'.format(evaluationId, msg.split('\n')[1].strip()))
    print(msg)
    return scores


def evaluate_labeling(dir_path, labeling: Dict[str, Dict[str, float]], key_path: str = None) \
        -> Dict[str, Dict[str, float]]:
    """
    labeling example : {'become.v.3': {'become.sense.1':3,'become.sense.5':17} ... }
    means instance become.v.3' is 17/20 in sense 'become.sense.5' and 3/20 in sense 'become.sense.1'
    :param key_path: write produced key to this file
    :param dir_path: SemEval dir
    :param labeling: instance id labeling
    :return: FNMI, FBC as calculated by SemEval provided code
    """

    def get_scores(gold_key, eval_key):
        ret = {}
        for metric, jar, column in [
            ('jaccard-index', os.path.join(dir_path, 'scoring/jaccard-index.jar'), 1),
            ('pos-tau', os.path.join(dir_path, 'scoring/positional-tau.jar'), 1),
            ('WNDC', os.path.join(dir_path, 'scoring/weighted-ndcg.jar'), 1),
            ('FNMI', os.path.join(dir_path, 'scoring/fuzzy-nmi.jar'), 1),
            ('FBC', os.path.join(dir_path, 'scoring/fuzzy-bcubed.jar'), 3),
        ]:
            res = subprocess.Popen(
                ['java', '-jar', jar, gold_key, eval_key], stdout=subprocess.PIPE).stdout.readlines()
            for line in res:
                line = line.decode().strip()
                if line.startswith('term'):
                    # columns = line.split('\t')
                    pass
                else:
                    split = line.split('\t')
                    if len(split) > column:
                        word = split[0]
                        # results = list(zip(columns[1:], map(float, split[1:])))
                        result = split[column]
                        if word not in ret:
                            ret[word] = {}
                        ret[word][metric] = float(result)

        return ret

    with tempfile.NamedTemporaryFile('wt') as fout:
        lines = []
        for instance_id, clusters_dict in labeling.items():
            clusters = sorted(clusters_dict.items(), key=lambda x: x[1])
            clusters_str = ' '.join(
                [('%s/%f' % (cluster_name, count)) for cluster_name, count in clusters])
            lemma_pos = instance_id.rsplit('.', 1)[0]
            lines.append('%s %s %s' % (lemma_pos, instance_id, clusters_str))
        fout.write('\n'.join(lines))
        fout.flush()

        scores = get_scores(os.path.join(
            dir_path, 'keys/gold/all.key'), fout.name)
        if key_path:
            with open(key_path, 'w', encoding="utf-8") as fout2:
                fout2.write('\n'.join(lines))
        return scores


if __name__ == '__main__':

    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    if input_path is None:
        raise Exception('missing inputPath')
    else:
        evaluateFromFile(input_path)
