import os
import csv

from time import sleep
import numpy as np
import pandas as pd
import pickle 
from tqdm.contrib.concurrent import process_map 
from tqdm.auto import tqdm
from homepage2vec.model import WebsiteClassifier

DATA_PATH = 'data'
model = WebsiteClassifier()

with open(os.path.join(DATA_PATH, 'website_labels.csv'), 'a', newline='') as csvfile:
	writer = csv.writer(csvfile)
	while True:
		website_files = sorted(os.scandir('websites'), key=lambda x: x.stat().st_mtime, reverse=False)
		if len(website_files) < 1000:
			sleep(10)
		for website_path in tqdm(website_files):
			website_page = pickle.load(open(website_path.path, 'rb'))
			try:
				scores, embeddings = model.predict(website_page)
				label = list(scores.keys())[np.argmax(list(scores.values()))]
			except Exception as e:
				label = ''
			writer.writerow([website_path.name, label])

			os.remove(website_path.path)
