Lil Cody - AI That Can Rap

### About
Lil Cody (short for Lil CodeAI) is a rap generating AI trained on songs of 'lil rappers' - rappers who have 'lil' in their names.

### How I Built It : 
A. Data Collection: 
    Process : Download lyrics using Genius API
    1. lil_rappers_wikipedia.txt - List of all the lil rappers (source : Wikipedia)
    2. lyrics_download.py - gets the lyrics for all songs for rapper in lil_rappers => appends it to dataframe.
    3. lyrics_mashed.txt - lyrics downloaded in lyrics_download.py all mashed together

B. GPT-2 Finetuning : 
    Process: GPT-2 is text generation AI. 

