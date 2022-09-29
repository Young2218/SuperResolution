import urllib.request
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
import time

SCROLL_PAUSE_SEC = 1
def scroll_down():
    global driver
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_SEC)
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            time.sleep(SCROLL_PAUSE_SEC)
            new_height = driver.execute_script("return document.body.scrollHeight")

            try:
                driver.find_element_by_class_name("mye4qd").click()
            except:

               if new_height == last_height:
                   break


        last_height = new_height



keyword = "urban image"
save_path = '/home/rnjbdya/Downloads/open/crwaling_data/'
chrome_driver_path='/home/rnjbdya/Downloads/chromedriver_linux64/chromedriver'
driver = webdriver.Chrome(chrome_driver_path)
url = 'https://www.google.com/search?q={}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjgwPKzqtXuAhWW62EKHRjtBvcQ_AUoAXoECBEQAw&biw=768&bih=712'.format(keyword)

driver.get(url)

time.sleep(1)

# scroll_down()

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
images = soup.find_all('img', attrs={'class':'rg_i Q4LuWd'})

print('number of img tags: ', len(images))

n = 1
keyword = keyword.replace(' ', '')
for i in images:
    try:
        imgUrl = i["src"]
    except:
        imgUrl = i["data-src"]
        
    with urllib.request.urlopen(imgUrl) as f:
        with open(save_path + keyword + str(n) + '.jpg', 'wb') as h:
            img = f.read()
            h.write(img)

    n += 1