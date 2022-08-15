from urllib.request import Request, urlopen, urlretrieve
import re
import sys
import time
from PIL import Image
import os



for page in range(100):
	for color in range(2):
		if color == 0:
			req = Request(f"https://nhentai.net/tag/full-color/?page={page}")
		else:
			req = Request(f"https://nhentai.net/?page={page}")
		req.add_header('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5')

		content = urlopen(req).read()
		x = re.findall("<a href=\"(.*?)\" class=\"cover\" style=", str(content))

		for website in range(1, len(x)):
			print(x[website])
			baselink = f"https://nhentai.net" + x[website]
			req = Request(baselink)
			req.add_header('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5')

			content = urlopen(req).read()
			pages = str(content).split("num_pages")[1].split(':')[1].split(',')[0]
			print(pages)

			for webNumber in range(int(pages)):

				req = Request(baselink + str(webNumber +1))
				req.add_header('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5')

				content = urlopen(req).read()

				w = re.findall("<img(.*?)>", str(content))
				link = w[1].strip().split('"')[1]
				print(link)
				if color == 0:
					urlretrieve(link, f"./data/hentai_A/{page}_{website}_{baselink.split('/')[4]}_{webNumber}.jpg")
				else:
					if "full color" not in str(content):
						urlretrieve(link, f"./data/hentai_B/{page}_{website}_{baselink.split('/')[4]}_{webNumber}.jpg")
				time.sleep( 3 )


	
