import lyricsgenius
import pandas as pd 

CLIENT_ACCESS_TOKEN = 'bL4Qr_Z8BkqTWEzYXUOTHaIpyweE3bVky4UcrtnJJinVMbJm6xIE3A_6iLofD2Ep'

#Get all the lil rappers from lil rapper list on Wikipedia
lil_rappers_file = open("lil_rappers_wikipedia.txt","r")
rapper_list = lil_rappers_file.readlines()
rapper_list = [rapper.split(',')[0] for rapper in rapper_list]
lil_rappers_file.close()

genius = lyricsgenius.Genius(CLIENT_ACCESS_TOKEN)

lyrics_df = pd.DataFrame()
lyrics_mash_file = open('lyrics_mashed.txt', 'a+')

for rapper_name in rapper_list: 
    rapper = genius.search_artist(rapper_name, max_songs=10)
    try: 
        for song in rapper.songs: 
            try:
                lyrics_mash_file.write(song.lyrics)
            except: 
                continue
    except: 
        continue
lyrics_mash_file.close()
        



