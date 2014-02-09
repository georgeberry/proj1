#open file

biblefile = "/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj1/bible_corpus 2/kjbible.train"

bible = readall(open(biblefile, "r"))

d = Dict{ASCIIString, Int64}()
bigram_dict = Dict()


#sub for <WORDS>
#sub for 1:1
#get rid of newlines
#lowercase-ize
#add spaces before punctuation
function clean_up(bible::ASCIIString)
    bible = replace(bible, r"<.*>|[0-9]+:[0-9]+\s*", "")
    bible = replace(bible, r"\n+"," ")
    bible = lowercase(bible)
    bible = replace(bible, r"\w[^\w\s]", x -> string(x[1], " ", x[2]))
    b = split(bible, r" +")
    b = map(strip, b)
    return b
end


function make_unigrams(b::Array{ASCIIString}, d::Dict{ASCIIString, Int64})
    for word in b
        #temp = get(d::Dict{ASCIIString, Int64}, word, 0)
        #d[word] = temp + 1
        d[word] = get(d, word, 0) + 1
    end
    return d
end


function make_bigrams(b::Array{ASCIIString}, bigramdict::Dict)
    n1 = b[1]

    for word in b[2:end]
        n2 = word

        #num times we see n2 if we saw n1
        # {n2: {n1: 0, n1: 0...}
        #}


        bigramdict[n2] = get(bigramdict, n2, {n1=>0})
        bigramdict[n2][n1] += 1

        println(bigramdict)

        n1 = n2
        #bigramdict[b[word_num-1]] = get(bigramdict, )

        #key = (b[word_num-1],b[word_num])
        #bigramdict[key] = get(bigramdict, key, 0) + 1
    end
    return bigramdict
end


tic = time()
b = clean_up(bible)
println(time() - tic)

tic = time()
make_unigrams(b, d)
println(time() - tic)

tic = time()
make_bigrams(b, bigram_dict)
println(time() - tic)