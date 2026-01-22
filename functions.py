import os
import sys
from time import time
import yt_dlp
from pydub import AudioSegment

from requests_html import AsyncHTMLSession 
from bs4 import BeautifulSoup as bs
import re
import asyncio
import nest_asyncio # we need nested async because jupyter notebook has its own event loop running
nest_asyncio.apply()

import numpy as np
import pandas as pd
import pickle
import librosa 


from sklearn.preprocessing import StandardScaler
from scipy import spatial

# FUNCTION #1
def song_download_split(web_link,destination,sample_name,t_sec,duration = 10):
    """
    Download and store 10 second audio sample from YOUTUBE link.

    Parameters
    ----------
    web_link : string
        The youtube link that we get the audio sample from.
    destination : string
        The destination which we will store the sample .wav file
            >>>> f"{str(os.getcwd())}\datasets\AUDIO\{destination}\{sample_name}.wav"  
    sample_name : string
        Name of the sample for future references.
    t_sec : int
        Timestamp in seconds which is the moment 10 second sample begins.
    duration : int
        In seconds, how long the sample is going to be.

    Returns
    -------
    True

    """
    web_link = check_for_http(web_link)
    options = {
      'format': 'bestaudio/best',
      'extractaudio' : True,  # only keep the audio
      'audioformat' : "wav",  # convert to wav
      'outtmpl': 'temp.wav',    # '%(id)s' == name the file the ID of the video
      'noplaylist' : True,    # only download single song, not playlist
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([web_link])
        
    t1 = t_sec * 1000 #Works in milliseconds
    t2 = (t_sec +duration)* 1000
    AudioSegment.converter = r"ffmpeg/bin/ffmpeg.exe" #r"C:\ffmpeg\bin\ffmpeg.exe"
    newAudio = AudioSegment.from_file(r'temp.wav')
    newAudio = newAudio[t1:t2]
    
    sample_name = "".join(i for i in sample_name if i not in "\/:*?<>|") # DROP ILLEGAL CHARS FOR FILE NAMING
    newAudio.export(f"{str(os.getcwd())}\\datasets\\AUDIO\\{destination}\\{sample_name}.wav", format="wav")
    os.remove(r'temp.wav')
    return f"Downloaded:{destination}\\{sample_name}.wav"

# FUNCTION #2 
filter_genre_labels = lambda song_tags : {tag for tag in song_tags if tag in genre_labels}

# FUNCTION #3
filter_sub_genre_labels = lambda song_tags : { tag for tag in song_tags if tag in comprehensive_subgenres}

# FUNCTION #4
def get_youtube_data(video_url):
    """
    Parameters
    ----------
    video_url : string
        The youtube link that we scrape the data from.
        
    Returns
    -------
    title : string
        Title of the video.
        
    tags : set
        Every tag of the video in uppercase.(additionally splits and adds multiple word tags.)
        
    duration_in_seconds : int
        Duration of the video calculated from meta data.
    """
    video_url = check_for_http(video_url)
    soup = asyncio.run(create_soup(video_url))
    
    # GETTING THE INFORMATION WE NEED
    title = soup.find("meta", itemprop="name")["content"]
    
    # TAGS
    tags =[meta.attrs.get("content") for meta in soup.find_all("meta", {"property": "og:video:tag"})]
    for item in tags: # IF A TAG CONSISTS OF MULTIPLE WORDS, SPLIT AND ADD THEM AS NEW TAGS 
        item= item.split()
        if len(item) > 1:
            tags += item
    tags= {tag.upper() for tag in tags}
    
    # DURATION
    # duration is stored in a weird string
    duration_string = soup.find("meta",itemprop = "duration")["content"]
    list_dur= re.findall("\\d+",duration_string) #regex to find decimals in string  1,23,33
    
    coefs=[1,60,3600] # second,minute,hour >> in seconds 
    duration_in_seconds = 0
    for item,second in zip(list_dur[::-1],coefs[0:len(list_dur)]):
        duration_in_seconds += int(item)*second
        
    return (title, tags, duration_in_seconds)  #P4M43ST

# FUNCTION #5
def audio_features_extract(audio, sr = 22050):
    """
    >> audio_features_extract(audio, sr = 22050)
    
    Extract frequency domain features from audio sample using librosa.feature .
    Calculates the means and variances of the features below.
    
    rms
    zero_crossing_rate
    spectral_centroid
    spectral_rolloff
    spectral_bandwidth
    tempo
    20 x MFCC
    
    Parameters
    ----------
    audio : numpy.ndarray
        One dimensional array of numbers representing an .wav file.
        
    sr : int
        Sample rate of the audio file. Default is 22050.

    Returns
    ----------
       numeric_features : list
    
    ----------
    
    >>> len(numeric_features) 
    51
    """
    zero_crossings_rate= librosa.feature.zero_crossing_rate(audio)
    spectral_centroids = librosa.feature.spectral_centroid(audio, sr)
    spectral_rolloff   = librosa.feature.spectral_rolloff(audio, sr=sr)
    rms = librosa.feature.rms(audio,center=True)
    tempo = librosa.beat.tempo(audio,hop_length=256,sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    mfcc = librosa.feature.mfcc(audio, sr=sr)
    

    
    return  ([np.mean(rms),np.var(rms),np.mean(zero_crossings_rate), np.var(zero_crossings_rate), np.mean(spectral_centroids),  np.var(spectral_centroids), np.mean(spectral_rolloff),
              np.var(spectral_rolloff), np.mean(spectral_bandwidth),  np.var(spectral_bandwidth),tempo[0] ] + [np.mean(e) for e in mfcc] + [np.var(e) for e in mfcc])


# EXTRA FUNCTIONS
def check_for_http(url):
    if url[0:8] == "https://":
        new_url = url
    else:
        new_url = r"https://" + url
    return new_url

# Disable / enable prints to keep the output clean
temp_stdout = None
# Disable
def disablePrint():
    global temp_stdout
    temp_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
def enablePrint():
    global temp_stdout
    sys.stdout = temp_stdout
    
#sub-FUNCTION  of #4
async def create_soup(video_url):
    # init an HTML Session
    asession = AsyncHTMLSession()
    # get the html content
    response = await asession.get(video_url)
    # create bs object to parse HTML
    soup = bs(response.html.html, "html.parser")
    #soup.find_all("meta")
    return soup

genre_labels = pickle.load(open(r"datasets/genre_set","rb"))
comprehensive_subgenres = pickle.load(open(r"datasets/subgenre_set","rb"))


def get_cosine_similarity(row,scaled_new_data):
    similarity = 1 - spatial.distance.cosine(row, scaled_new_data)
    return similarity


def similar_songs(sample_features,df_to_compare):
    """
    Calculate the cosine similarity of a single vector to every
    vector in the given dataframe to compare.
    
    Parameters
    ----------
    sample_features : list
        One dimensional array of numbers features
        
    df_to_compare : <class 'pandas.core.frame.DataFrame'>
        Scaled features od every stored song.

    Returns
    ----------
       similar_songs : <class 'pandas.core.frame.DataFrame'>
    
    """
    features = df_to_compare.drop(['file_name', 'genre_label'], axis=1).values
    labels = df_to_compare.genre_label.values
    file_names = df_to_compare.file_name.values
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    scaled_new_data = scaler.transform(np.array(sample_features).reshape(1, -1))
    similarities = []
    for row in scaled_data:
        similarities.append(get_cosine_similarity(row,scaled_new_data))
    
    result = pd.DataFrame(columns = ["file_name",'genre','similarity'])
    result["file_name"] = file_names
    result["similarity"] = similarities
    result["genre"] = labels
    return result.sort_values(by='similarity',ascending=False).head()
    