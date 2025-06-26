import pandas as pd
from urllib import request
from gensim.models import word2vec

data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')

lines = data.read().decode('utf-8').split('\n')[2:]

playlists = [s.rstrip().split() for s in lines if len(s.split())>1]

songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')

songs_file = songs_file.read().decode('utf-8').split('\n')

songs = [s.rstrip().split('\t') for s in songs_file]
songs_df = pd.DataFrame.from_records(songs, columns=['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')
# print(songs_df.head())

model = word2vec.Word2Vec(
    playlists,
    vector_size=32,
    window=20,
    negative=50,
    min_count=1,
    workers=4,
)

song_id = 2172
res = model.wv.most_similar(positive=str(2172))

print(res)

for item, pob in res:
    print(songs_df.iloc[int(item)])



