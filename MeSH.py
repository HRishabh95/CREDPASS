import requests
from bs4 import BeautifulSoup
query='diet'

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
base_url = 'https://meshb.nlm.nih.gov'
quer_url = f'''https://meshb.nlm.nih.gov/search?searchMethod=FullWord&searchInField=termDescriptor&sort=&size=20&searchType=exactMatch&from=0&q={query}'''
r = requests.get(quer_url, headers=headers)
soup = BeautifulSoup(r.content, 'lxml')
result = soup.findAll('div', {'id': "searchResults"})
