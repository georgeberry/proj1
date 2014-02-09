import re, time

#open file
biblefile = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj1/bible_corpus 2/kjbible.train'

with open(biblefile, 'rb') as f:
    bible = f.read()

#sub for <WORDS>
#sub for 1:1
#get rid of newlines
#lowercase-ize

def clean_up(bible):
    bible = re.sub(r'<.*>','',bible)
    bible = re.sub(r'[0-9]+:[0-9]+\s*','', bible)
    bible = re.sub(r'\n+',' ', bible)
    bible = bible.lower().strip()
    return bible

def make_unigrams(b):
    b = re.split(r' +', b)

    d = {}

    for word in b:
        try:
            d[word] += 1
        except:
            d[word] = 1
    return d

def make_bigrams(b):
    bgd = {}

    for word in range(1, len(b)):
        try:
            bgd[(b[word-1], b[word])] += 1
        except:
            bgd[(b[word-1], b[word])] = 1
    return bgd

tic = time.time()
bible = clean_up(bible)
print time.time() - tic

tic = time.time()
a = make_unigrams(bible)
print time.time() - tic

tic = time.time()
b = make_bigrams(bible)
print time.time() - tic