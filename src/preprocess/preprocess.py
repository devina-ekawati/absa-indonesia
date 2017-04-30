import sys, re, os

file_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(file_path, os.path.pardir))

sys.path.append(os.path.join(project_path, "../lib/InaNLP.jar"))
sys.path.append(os.path.join(project_path, "../lib/ipostagger.jar"))

from IndonesianNLP import IndonesianSentenceFormalization
from IndonesianNLP import IndonesianPOSTagger
from IndonesianNLP import IndonesianSentenceDetector

class Preprocess:
    def splitIntoSentence(self, sentences):
        detector = IndonesianSentenceDetector()
        return detector.splitSentence(sentences);

    def formalizeSentence(self, sentence):
        formalizer = IndonesianSentenceFormalization()
        return formalizer.formalizeSentence(sentence)

    def deleteStopWord(self, sentence):
        formalizer = IndonesianSentenceFormalization()
        formalizer.initStopword();
        return formalizer.deleteStopword(sentence)

    def posTagger(self, sentence):
        InaPosTagger = IndonesianPOSTagger()
        return InaPosTagger.doPOSTag(sentence)


if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print("Syntax: ", sys.argv[0], "<input file> <output file>")
        sys.exit(1)

    preprocess = Preprocess()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    regex = re.compile('[^0-9a-zA-Z]+')

    lines = []
    f = open(input_file, "r")
    for line in f:
        line = line.rstrip()
        lines.append(line)
    f.close()

    f = open(output_file, "w")
    for line in lines:
        sentences = preprocess.splitIntoSentence(line)
        for sentence in sentences:
            # f.write(preprocess.formalizeSentence(sentence).lower() + " ")
            f.write(regex.sub(' ', preprocess.formalizeSentence(sentence).lower() + " "))
            f.write("\n")
    f.close()

    # sentences = "Halo semua. selamat siang."
    # sentences = preprocess.splitIntoSentence(sentences)
    # for sentence in sentences:
    # 	print sentence

    # sentence = "kata2nya 4ku donk loecoe bangedh gt"
    # sentence =  preprocess.formalizeSentence(sentence)
    # print sentence
    # sentence = preprocess.deleteStopWord(sentence)
    # print sentence
    # preprocess.posTagger(sentence)
