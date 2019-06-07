from urllib.request import Request, urlopen, urlretrieve
import re
import sys

baselink = sys.argv[1]
req = Request(baselink)
req.add_header('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5')

content = urlopen(req).read()
pages = str(content).split("num_pages")[1].split(':')[1].split(',')[0]
print(pages)

for webNumber in range(int(pages)):
	req = Request(baselink + str(webNumber +1))
	req.add_header('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5')

	content = urlopen(req).read()

	x = re.findall("<img(.*?)>", str(content))
	link = x[1].strip().split('"')[1]
	print(link)
	urlretrieve(link, "./data/hentai/" + baselink.split('/')[4]+str(webNumber)+".jpg")

