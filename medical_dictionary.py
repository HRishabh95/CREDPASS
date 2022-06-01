import requests
from bs4 import BeautifulSoup
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

def get_alphalinks(soup):
    alphalinks = soup.find_all('div', {'class': 'alphalinks'})
    a_links=alphalinks[0].find_all('a')
    a_link_list=["https://www.merriam-webster.com"+a_link.get('href') for a_link in a_links]
    return a_link_list

def get_terms(soup):
    names = soup.find_all('div', {'class': 'mw-grid-table-list'})[0]
    names = names.find_all('a')
    return [name.get_text().lower() for name in names]

r=requests.get("https://www.merriam-webster.com/browse/medical",headers=headers)
soup = BeautifulSoup(r.content)
a_link_lists=get_alphalinks(soup)
terms=[]
for a_link in a_link_lists:
    r = requests.get(a_link, headers=headers)
    soup = BeautifulSoup(r.content)
    terms+=get_terms(soup)
    while soup.find_all('link',{'rel':'next'}):
        r=requests.get(soup.find_all('link',{'rel':'next'})[0].get('href'),headers=headers)
        soup = BeautifulSoup(r.content)
        terms+=get_terms(soup)


medical_words=open('wordlist.txt').readlines()
medical_words=[i.lower().replace("\n","") for i in medical_words]
new_medical_terms=terms

combined_set=set(terms+medical_words)
combined=list(combined_set)

f=open('medical_term.txt','w')
for i in combined:
    f.write(f'''{i}\n''')
f.close()
