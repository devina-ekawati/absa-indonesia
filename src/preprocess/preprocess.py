import sys
sys.path.append("../lib/InaNLP.jar")
sys.path.append("../lib/ipostagger.jar")

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
	preprocess = Preprocess()
	sentences = "Halo semua. selamat siang."
	sentences = preprocess.splitIntoSentence(sentences)
	for sentence in sentences:
		print sentence

	sentence = "kata2nya 4ku donk loecoe bangedh gt"
	sentence =  preprocess.formalizeSentence(sentence)
	print sentence
	sentence = preprocess.deleteStopWord(sentence)
	print sentence
	preprocess.posTagger(sentence)