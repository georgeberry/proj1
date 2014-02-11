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
    #remove xml tags
    bible = re.sub(r'<.*>','',bible)
    #replace verse numbers (beginning of sentence) with <s> tag
    bible = re.sub(r'[0-9]+:[0-9]+\s*','', bible)
    #replace newlines with spaces (we do a space split below)
    bible = re.sub(r'[\n]+',' ', bible)
    #replace punctuation with a space then that punctuation
    bible = re.sub(r"[^\w\s\']+", " \g<0>", bible)
    bible = re.sub(r'\.', '. </s> <s>', bible)
    bible = bible.lower().strip()
    bible = re.split(r' +', bible)
    bible.append('</s>') #just need this
    #we automatically put <s> at the start with make_ngrams function

    return bible


def word_or_start_token(text_list, token_num):
    if token_num < 0:
        return '<s>'
    else:
        return text_list[token_num]


##make ngrams
def make_ngrams(text_list, n):
    n_gram_dict = {}
    range_n = range((n-1), -1, -1)

    #initialize with repeated <s> at the beginning
    for token_num in range(len(text_list)):
        n_gram = []

        #count backwards from n-1 to 0
        #gives the word at position token_num, along with the previous n-1 words
        for countdown in range_n:
            try:
                n_gram.append(text_list[token_num - countdown])
            except:
                n_gram.append('<s>')
            #n_gram.append(word_or_start_token(text_list, token_num - countdown))

        #for unigrams, key the word directly in the dictionary rather than in a tuple
        if n == 1:
            try:
                n_gram_dict[n_gram[0]] += 1
            except:
                n_gram_dict[n_gram[0]] = 1

        #for n > 1, tuples are dict keys
        else:
            try:
                n_gram_dict[tuple(n_gram)] += 1
            except:
                n_gram_dict[tuple(n_gram)] = 1
    return n_gram_dict


bible = clean_up(bible)

tic = time.time()
aa = make_ngrams(bible,1)
print time.time() - tic

tic = time.time()
bb = make_ngrams(bible,2)
print time.time() - tic

tic = time.time()
cc = make_ngrams(bible,3)
print time.time() - tic

tic = time.time()
cc = make_ngrams(bible,4)
print time.time() - tic

