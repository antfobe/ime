#!/usr/bin/python

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import time, re, random, json, unicodedata, copy

def setup_driver():
    driver = webdriver.Firefox()
    ## reducing implicit wait - 30s is to long to wait for a non-registered author
    driver.implicitly_wait(10)
    driver.base_url = "https://scholar.google.com/"
    driver.verificationErrors = []
    driver.accept_next_alert = True
    driver.get("https://scholar.google.com/")
    ## click 'scholar in english' link
    driver.find_element_by_id("gs_hp_eng").find_element_by_tag_name("a").click()
    time.sleep(random.uniform(3, 7))
    driver.find_element_by_id("gs_hdr_tsi").clear()
    time.sleep(random.uniform(3, 7))
    return driver

def is_element_present(element, how, what):
    try: element.find_element(by=how, value=what)
    except NoSuchElementException as e: return False
    return True

def driver_get_author_info(driver, author_list):
    author_dict = {}
    author_dictlist = []
    author_info = []
    for author_name in author_list:
        driver.find_element_by_id("gs_hdr_tsi").clear()
        driver.find_element_by_id("gs_hdr_tsi").send_keys(author_name)
        driver.find_element_by_id("gs_hdr_frm").submit()
        time.sleep(random.uniform(3, 7))
        if is_element_present(driver, By.TAG_NAME, "h4"):
            result = driver.find_element_by_tag_name("h4")
            result = result.find_element_by_xpath("..")
            for elemnt in result.find_elements_by_tag_name("div"):
                author_info.append(elemnt.get_attribute("innerHTML"))
            author_dict["name"] = author_name
            author_dict["info"] = author_info
            author_dictlist.append(copy.deepcopy(author_dict))
            author_info = []
        else:
            author_dict["name"] = author_name
            author_dict["info"] = "Info Unavailable"
            author_dictlist.append(copy.deepcopy(author_dict))
        print(author_dictlist)
    return author_dictlist

author_list = []
with open("bibtex_refs.json", "r") as file:
    article_data = json.load(file)
    for idx, item in enumerate(article_data):
        data = copy.deepcopy(json.loads(article_data[idx]))
        print(data)
        for name in data["author"].split(" and "):
            print("[" + name + "]\n")
            if len(name.split(',')) > 1:
                initials = "".join([i[0].upper() for i in name.split(',')[1].split()])
                item = unicodedata.normalize(
                            'NFKD',
                            (initials + " " + name.split(',')[0]).replace('-', ' ')
                ).encode('ascii', 'ignore').decode()
                author_list.append(item)
            else:
                initials = "".join([i[0].upper() for i in " ".join(name.split(' ')[0:-1]).split()])
                item = unicodedata.normalize(
                            'NFKD',
                            (initials + name.split(' ')[-1]).replace('-', ' ')
                ).encode('ascii', 'ignore').decode()
                author_list.append(item)
author_list = list(set(author_list))
driver = setup_driver()
author_infos = driver_get_author_info(driver, author_list)
driver.close()
with open("authors_info.json", "w") as f:
    f.write(json.dumps(author_infos, sort_keys=True, indent=4, separators=(',', ': ')))
