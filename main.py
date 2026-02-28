import nltk
from nltk.corpus import brown
from implement.implement import Viterbi


def main():
    tagged_sents = brown.tagged_sents(tagset='universal')
    tagged_words = brown.tagged_words(tagset='universal')
    
    v = Viterbi(tagged_sents, tagged_words)

    v.init()

    # sentence = ["I", "love", "cats", "."]
    # sentence = ["The", "old", "man", "the", "boat", "."]
    # sentence = ["The", "complex", "houses", "married", "and", "single", "soldiers", "."]
    sentence = ["Time", "flies", "like", "an", "arrow", ";", "fruit", "flies", "like", "a", "banana", "."]

    path = v.find_tags(sentence, detailed=True)

    print(path)

if __name__ == "__main__":
    nltk.download('brown')
    nltk.download('universal_tagset')
    main()
