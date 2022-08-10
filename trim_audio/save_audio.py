import re
import pytz
import requests
import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin

surah_alaq_link = "https://qurancentral.com/surah/096-al-alaq/"

r = requests.get(surah_alaq_link)
surah_alaq_html = r.text

soup = BeautifulSoup(surah_alaq_html, "html.parser")

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36','charset': 'utf-8'}

table = soup.find('table' , {"id": "audio-episodes"})
tr = table.find('tbody').findAll('tr')  
for t in tr:
    url= t.find('td')['data-episodeurl']
    url = "https://media.blubrry.com/muslim_central_quran/podcasts.qurancentral.com/"+url
    print("Downloading {}".format(url))

    with open("surah_alaq/"+url.split("/")[-2]+".mp3", "wb") as f_out:
        f_out.write(requests.get(url, headers=headers).content)


