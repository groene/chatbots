import os
import pandas as pd
import pickle 
from tqdm.contrib.concurrent import process_map  # or thread_map

from homepage2vec.model import WebsiteClassifier

model = WebsiteClassifier(device='cpu')

output_dir_websites = 'websites'
websites = pd.read_csv('data/url_counts.csv')

def download_website(website_url):
	website = model.fetch_website(website_url)
	pickle.dump(website, open(os.path.join(output_dir_websites, website_url + '.p'), 'wb'))

if not os.path.exists('last_idx.txt'):
	last_idx = 0
	f = open('last_idx.txt', 'w')
	f.write(str(last_idx))
	f.close()
else:
	last_idx = int(open('last_idx.txt', 'r').read())


while True:
	while len(os.listdir(output_dir_websites)) < 10000:
		chunk = websites.iloc[last_idx:last_idx+1000]
		
		r = process_map(download_website, chunk.url, max_workers=15)

		last_idx += 1000
		f = open('last_idx.txt', 'w')
		f.write(str(last_idx))
		f.close()


