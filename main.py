import nltk
from nltk.corpus import treebank
from implement.implement import Viterbi


def main():
    tagged_sents = treebank.tagged_sents()
    tagged_words = treebank.tagged_words()
    
    v = Viterbi(tagged_sents, tagged_words)

    v.init()

    sentence = ["I", "love", "cats", "."]

    path = v.find_tags(sentence)

    print(path)

if __name__ == "__main__":
    nltk.download('treebank')
    nltk.download('tagsets_json')
    main()
