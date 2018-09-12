#!/usr/bin/python

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re, random, bibtexparser, json

#search_query = scholarly.search_pubs_query('Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture')
#article = next(search_query).fill()
#for author in article.bib['author'].split('and'):
#    query = scholarly.search_author(author)
#    try:
#        this = next(query)
#        print(article + ',' + author + ',' + this.affiliation + ',')
#    except StopIteration:
#        print(article + ',' + author + ',Unknown affiliation,')

def setup_driver(driver):
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

def driver_item_bylink(driver, link, tag):
    driver.execute_script('''window.open("''' + link + '''", "_blank")''')
    driver.switch_to.window(driver.window_handles[len(driver.window_handles) - 1])
    element = driver.find_element_by_tag_name(tag).get_attribute
    driver.execute_script('''window.close()''')
    return element

def readref_tree(article_link, citation, article_list):
    print("At: ["+article_link+"]")
    parser = bibtexparser.bparser.BibTexParser()
    parser.customization = bibtexparser.customization.convert_to_unicode
    ref_list = []
    ## im assuming if it knows the citing article, it has its name
    #article = scholarly.search_pubs_query(article_name)
    driver.find_element_by_id("gs_hdr_tsi").click()
    time.sleep(random.uniform(3, 7))
    driver.find_element_by_id("gs_hdr_tsi").clear()
    driver.find_element_by_id("gs_hdr_tsi").send_keys(article_name)
    driver.find_element_by_id("gs_hdr_frm").submit()
    time.sleep(random.uniform(3, 7))
    try:
        for ref_link in driver.find_elements_by_link_text("Importe para o BibTeX").get_attribute("href"):
            ref.append(bibtexparser.loads(driver_item_bylink(driver, ref_link, "pre").get_attribute("innerHTML"), parser=pareser).get_entry_list()[0])
            ref["cites"] = citation
            with open("bibtex_refs.txt", "a") as f:
                f.write(json.dumps(ref) + "\n")
            ref_list.append(ref)
            article_list.append(citation)

        link_tags = driver.find_elements_by_partial_link_text("Citado por ")

        for tag in link_tags:
            link = tag.get_attribute("href")
            driver.get(link)
            if ref["Title"] not in article_list:# check this field


        if article.get_citedby() is not None and not any(article.get_citedby() in c for c in found['citations']):
            found['citations'].append(article.get_citedby())
            for ref in article.get_citedby():
                readref_tree(ref.bib['title'], article_name, found)
        else:
            for author in article.bib['author'].split('and'):
                if not any(author in a for a in found['authors']):
                    found['authors'].append(author)
                    query = scholarly.search_author(author)
                    try:
                        this = next(query)
                        with open('refsout.csv', 'a') as f:
                            f.write(article_name + ',' + author + ',' + this.affiliation + ',' + citation + '\n')
                    except StopIteration:
                        with open('refsout.csv', 'a') as f:
                            f.write(article_name + ',' + author + ',Unknown affiliation,' + citation + '\n')
    except:
        with open('refsout.csv', 'a') as f:
            f.write(article_name + ',Request not completed,Unknown affiliation,Request not completed' + '\n')

memdict = {'authors': [], 'citations': []}
readref_tree('Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture', '', memdict)
