# -*- coding: utf-8 -*-
import urllib
import urllib.request
from bs4 import BeautifulSoup

i = 9
while i < 13:
    print("crawl" + str(i) + "page")
    url = "http://gb.weather.gov.hk/cis/dailyExtract_uc.htm?y=2015&m=" + str(i)
    print(url)
    content = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(content, "html.parser")
    print(soup.title.get_text())

    num = soup.find_all("th", class_="td_normal_class")
    for n in num:
        con = n.get_text()
        num = con.splitlines()
        print(num[1], num[2], num[3], num[4], num[5])
    i = i + 1
