from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup, NavigableString
from typing import Dict

RAW_DATA = "SemEval-2013-Task-13-test-data/contexts/senseval2-format/semeval-2013-task-13-test-data.senseval2.xml"

# POS tagging from SemEval to WordNet
pos_to_wnpos = {
    'v': wn.VERB,
    'n': wn.NOUN,
    'j': wn.ADJ
}


class Lexelt:
    def __init__(self, lemma_pos: str):
        self.lemma, pos = lemma_pos.split('.')
        self.pos = pos_to_wnpos[pos]
        self.instances = {}
        self.max_sentence_len = 0

    def addInstance(self, token: str, num: str, context: str):
        self.instances[num] = {"token": token, "context": context}
        self.max_sentence_len = max(len(context), self.max_sentence_len)


def loadSenseval2Format(filename: str = RAW_DATA) -> Dict[str, Lexelt]:
    Dataset = {}
    with open(filename, 'r') as text_raw:
        raw_xml = text_raw.read()

    xml = BeautifulSoup(raw_xml, features="html.parser")

    for lexelt in xml.corpus.findAll("lexelt"):
        item = lexelt['item']
        currentLexelt = Lexelt(item)

        for instance in lexelt.findAll("instance"):
            num = instance["id"][-1]
            token = instance.head.text
            context = instance.context.text
            currentLexelt.addInstance(token, num, context)

        Dataset[item] = currentLexelt

    return Dataset


def main():
    # Test parsing
    Dataset = loadSenseval2Format()
    print(Dataset["become.v"].lemma)
    print(Dataset["become.v"].pos)
    print(Dataset["become.v"].instances["1"])
    print(Dataset["become.v"].max_sentence_len)


if __name__ == "__main__":
    main()
