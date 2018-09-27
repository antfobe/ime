#!/usr/bin/python

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re, random, bibtexparser, json, unicodedata, os, sys, argparse

def setup_driver(article_name):
    driver = webdriver.Firefox()
    driver.implicitly_wait(30)
    driver.base_url = "https://scholar.google.com.br/"
    driver.verificationErrors = []
    driver.accept_next_alert = True
    driver.get("https://scholar.google.com.br/")
    ## click 'scholar in english' link
    driver.find_element_by_id("gs_hp_eng").find_element_by_tag_name("a").click()
    time.sleep(random.uniform(3, 7))
    driver.find_element_by_id("gs_hdr_tsi").click()
    driver.find_element_by_id("gs_hdr_tsi").clear()
    driver.find_element_by_id("gs_hdr_tsi").send_keys(article_name)
    driver.find_element_by_id("gs_hdr_frm").submit()
    time.sleep(random.uniform(3, 7))
    ###driver.find_element_by_xpath("(.//*[normalize-space(text()) and normalize-space(.)='Sign In'])[1]/following::span[1]").click()
    driver.find_element_by_id("gs_hdr_mnu").click()
    ###driver.find_element_by_xpath(u"(.//*[normalize-space(text()) and normalize-space(.)='Advanced Search'])[2]/following::span[2]").click()
    driver.find_element_by_id("gs_hdr_drw_bs").click()
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
    driver.find_element_by_id("gs_hdr_tsi").send_keys(article_name)
    driver.find_element_by_id("gs_hdr_frm").submit()
    time.sleep(random.uniform(3, 7))
    return driver

def driver_item_bylink(driver, link, tag, attr):
    return_url = driver.current_url
    driver.get(link)
    attribute = driver.find_element_by_tag_name(tag).get_attribute(attr)
    time.sleep(random.uniform(1, 3))
    driver.get(return_url)
    return attribute

def is_element_present(element, how, what):
    try: element.find_element(by=how, value=what)
    except NoSuchElementException as e: return False
    return True

def readref_tree(article_link, citation, article_list, bibtex_parser, verbose):
    if verbose:
        print("At: ["+article_link+"]")
    ## im assuming if it knows the citing article, it has its name

    driver.get(article_link)
    link_list = []
    bib_link_list = []
    refs = []
    next_results_link = driver.current_url
    while next_results_link is not None:
        for result_child in driver.find_elements_by_tag_name("h3"):
            result = result_child.find_element_by_xpath("..")
            bib_link_list.append(result.find_element_by_link_text("Import into BibTeX").get_attribute("href"))
        for bib_link in bib_link_list:
            bib_string = driver_item_bylink(driver, bib_link, "pre", "innerHTML")
            refs.append(bibtexparser.loads(bib_string, parser=bibtex_parser).get_entry_list()[-1])
        driver.get(article_link)
        try:
            ## need to check if there are more results than page can fit > 20
            ## ...and add cross-refs
            for idx, result_child in enumerate(driver.find_elements_by_tag_name("h3")):
                result = result_child.find_element_by_xpath("..")
                ref = refs[idx]
                if verbose:
                    print(str(ref) + '\nidx: ' + str(idx) + '\n')
                ref["title"] = unicodedata.normalize('NFKD', ref['title']).encode('ascii', 'ignore').decode().replace('\'', '')
                if ref["title"] not in article_list:
                    article_list.append(ref["title"])
                    ref["cites"] = citation
                    with open("bibtex_refs.txt", "a") as f:
                        f.write(json.dumps(ref) + "\n")
                    if is_element_present(result, By.PARTIAL_LINK_TEXT, "Cited by "):
                        link = driver.find_element_by_partial_link_text("Cited by ").get_attribute("href")
                        link_list.append({'citation':ref["title"], 'link':link})
            for linkd in link_list:
                readref_tree(linkd['link'], linkd['citation'], article_list, bibtex_parser, verbose)
            if is_element_present(driver, By.CLASS_NAME, "gs_ico_nav_next") is not False:
                next_results_link = driver.find_element_by_class_name("gs_ico_nav_next").find_element_by_xpath("..").get_attribute("href")
                driver.get(next_results_link)
            else:
                next_results_link = None

        except NoSuchElementException as e:
            sys.stderr.write("Request to\t[" + article_link + "]\tFailed ...\n")


argparser = argparse.ArgumentParser(description='Get an article reference tree from Google Scholar (BibTex links)')
argparser.add_argument('-a', '--aname', required=True, dest='entry_article', type=str, help='article/paper name/title (preferably normalized in NFKD)')
argparser.add_argument('-o', '--output', dest='out', type=str, help='output to [<filename>.json], if no filename is given, outputs to stdout')
argparser.add_argument('-v', '--verbose', dest='verb', help='displays execution messages to stdout', action="store_true")
args = argparser.parse_args()

article_list = []
parser = bibtexparser.bparser.BibTexParser()
parser.customization = bibtexparser.customization.convert_to_unicode
driver = setup_driver(args.entry_article)
readref_tree(driver.current_url, '', article_list, parser, args.verb)
driver.close()
with open("bibtex_refs.txt", "r") as f:
    lines = [line.rstrip('\n') for line in f]
os.remove("bibtex_refs.txt")
if args.out is not None:
    with open(args.out + ".json", "w") as f:
        f.write(json.dumps(lines, sort_keys=True, indent=4, separators=(',', ': ')))
else:
    sys.stdout.write(json.dumps(lines, sort_keys=True, indent=4, separators=(',', ': ')) + "\n")
