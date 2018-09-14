#!/usr/bin/python

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re, random, bibtexparser, json, unicodedata, os

#search_query = scholarly.search_pubs_query('Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture')
#article = next(search_query).fill()
#for author in article.bib['author'].split('and'):
#    query = scholarly.search_author(author)
#    try:
#        this = next(query)
#        print(article + ',' + author + ',' + this.affiliation + ',')
#    except StopIteration:
#        print(article + ',' + author + ',Unknown affiliation,')

def setup_driver():
    driver = webdriver.Firefox()
    driver.implicitly_wait(30)
    driver.base_url = "https://scholar.google.com.br/"
    driver.verificationErrors = []
    driver.accept_next_alert = True
    driver.get("https://scholar.google.com.br/")
    driver.find_element_by_id("gs_hdr_tsi").click()
    time.sleep(random.uniform(3, 7))
    driver.find_element_by_id("gs_hdr_tsi").clear()
    driver.find_element_by_id("gs_hdr_tsi").send_keys("Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture")
    driver.find_element_by_id("gs_hdr_frm").submit()
    time.sleep(random.uniform(3, 7))
    driver.find_element_by_xpath("(.//*[normalize-space(text()) and normalize-space(.)='Fazer login'])[1]/following::span[1]").click()
    driver.find_element_by_xpath(u"(.//*[normalize-space(text()) and normalize-space(.)='Pesquisa avançada'])[2]/following::span[2]").click()
    time.sleep(random.uniform(3, 7))
    ## set search results to 20 per page
    driver.find_element_by_id("gs_num-b").click()
    driver.find_element_by_link_text("20").click()
    time.sleep(random.uniform(3, 7))
    ## set bibtex export links
    driver.find_element_by_id("gs_settings_import_some").click()
    time.sleep(random.uniform(3, 7))
    driver.find_element_by_name("save").click()
    time.sleep(random.uniform(3, 7))
    ## search article name in scholar's search bar
    driver.find_element_by_id("gs_hdr_tsi").click()
    time.sleep(random.uniform(3, 7))
    driver.find_element_by_id("gs_hdr_tsi").clear()
    driver.find_element_by_id("gs_hdr_tsi").send_keys("Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture")
    driver.find_element_by_id("gs_hdr_frm").submit()
    time.sleep(random.uniform(3, 7))
    return driver

def driver_item_bylink(driver, link, tag, attr):
    return_url = driver.current_url
    driver.get(link)
    attribute = driver.find_element_by_tag_name(tag).get_attribute(attr)
    time.sleep(random.uniform(3, 7))
    driver.get(return_url)
    return attribute

def is_element_present(element, how, what):
    try: element.find_element(by=how, value=what)
    except NoSuchElementException as e: return False
    return True

def readref_tree(article_link, citation, article_list, bibtex_parser):
    print("At: ["+article_link+"]")
    ## im assuming if it knows the citing article, it has its name
    #article = scholarly.search_pubs_query(article_name)
    #driver.find_element_by_id("gs_hdr_tsi").click()
    #time.sleep(random.uniform(3, 7))
    #driver.find_element_by_id("gs_hdr_tsi").clear()
    #driver.find_element_by_id("gs_hdr_tsi").send_keys(article_name)
    #driver.find_element_by_id("gs_hdr_frm").submit()
    #time.sleep(random.uniform(3, 7))

    driver.get(article_link)
    link_list = []
    bib_link_list = []
    refs = []
    for result_child in driver.find_elements_by_tag_name("h3"):
        result = result_child.find_element_by_xpath("..")
        #print(str(link_list) + '\n')
        bib_link_list.append(result.find_element_by_link_text("Importe para o BibTeX").get_attribute("href"))
    for bib_link in bib_link_list:
        bib_string = driver_item_bylink(driver, bib_link, "pre", "innerHTML")
        #print(str(bib_link_list) + '\n')
        refs.append(bibtexparser.loads(bib_string, parser=bibtex_parser).get_entry_list()[-1])

    driver.get(article_link)
    try:
        ## need to check if there are more results than page can fit > 20
        for idx, result_child in enumerate(driver.find_elements_by_tag_name("h3")):
            result = result_child.find_element_by_xpath("..")
            ref = refs[idx]
            print(str(ref) + '\nidx: ' + str(idx) + '\n')
            ref["title"] = unicodedata.normalize('NFKD',ref['title']).encode('ascii', 'ignore').decode().replace('\'','')
            if ref["title"] not in article_list:
                article_list.append(ref["title"])
                ref["cites"] = citation
                with open("bibtex_refs.txt", "a") as f:
                    f.write(json.dumps(ref) + "\n")
                if is_element_present(result,By.PARTIAL_LINK_TEXT, "Citado por "):
                    link = driver.find_element_by_partial_link_text("Citado por ").get_attribute("href")
                    link_list.append({'citation':ref["title"], 'link':link})
        for linkd in link_list:
            readref_tree(linkd['link'], linkd['citation'], article_list, bibtex_parser)

    except NoSuchElementException as e:
        print("Request to\t[" + article_link + "]\tFailed ...\n")

article_list = []
parser = bibtexparser.bparser.BibTexParser()
parser.customization = bibtexparser.customization.convert_to_unicode
driver = setup_driver()
readref_tree(driver.current_url, '', article_list, parser)
driver.close()
with open("bibtex_refs.txt", "r") as f:
    lines = [line.rstrip('\n') for line in f]
os.remove("bibtex_refs.txt")
with open("bibtex_refs.json", "w") as f:
    f.write(json.dumps(lines, sort_keys=True, indent=4, separators=(',', ': ')))
