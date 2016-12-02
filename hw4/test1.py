import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans,MiniBatchKMeans
import sys,re

def main(argv):
	input_buffer = []
	test_buffer = []
	stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
	with open(argv[0]+'title_StackOverflow.txt') as textFile:
		for line in textFile:
			line = line.strip().split('\n')
			input_buffer.extend(line)
	with open(argv[0]+'docs.txt') as textFile:
		for line in textFile:
			line = line.strip()
			line = re.split('\W+', line)
			test_buffer.extend(line)
	test_buffer = filter(None,test_buffer)
	
	#print test.shape()
	#print input_buffer
	X = TfidfVectorizer(stop_words = stopwords)
	tfidf = X.fit_transform(test_buffer)
	Y = X.transform(input_buffer)
	#kmeans = KMeans(n_clusters=20, random_state=0, max_iter = 300).fit(tfidf)
	
	svd = TruncatedSVD(n_components = 20)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)
	Y = lsa.fit_transform(Y)
	print Y.shape
	#km = KMeans(n_clusters=20, init='k-means++', max_iter=5000, n_init=1).fit(X)
	km = MiniBatchKMeans(n_clusters=100, init='k-means++', max_iter=300, batch_size= 100, n_init=50).fit(Y)
	predict = km.labels_

	#testing data
	test_index = []
	with open(argv[0]+'check_index.csv') as textFile:
		next(textFile)
		for line in textFile:
			test_buffer = line.strip().split(',')
			for i in range (1):
				test_buffer.pop(0)
			test_index.append(test_buffer)
	test_index = np.array(test_index)
	test_index = test_index.astype(np.int)
	print test_index.shape
	#output
	file = open(argv[1],'w')
	file.write('ID,Ans\n')
	for i in range(5000000):
		file.write(str(i)+',')
		a = test_index[i][0]
		b = test_index[i][1]
		if predict[a] == predict[b] :
			file.write('1')
		else :
			file.write('0')
		file.write('\n')
	file.close()
	

if __name__ == '__main__':
	main(sys.argv[1:])