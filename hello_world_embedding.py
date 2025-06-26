import gensim.downloader as api

model = api.load('glove-wiki-gigaword-50')

res = model.most_similar([model['king']], topn=10)

print(res)