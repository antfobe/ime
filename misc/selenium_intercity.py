# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re, random

class AppDynamicsJob(unittest.TestCase):
    def setUp(self):
        # AppDynamics will automatically override this web driver
        # as documented in https://docs.appdynamics.com/display/PRO44/Write+Your+First+Script
        self.driver = webdriver.Firefox()
        self.driver.implicitly_wait(30)
        self.base_url = "https://www.katalon.com/"
        self.verificationErrors = []
        self.accept_next_alert = True

    def test_app_dynamics_job(self):
        driver = self.driver
        driver.get("https://scholar.google.com.br/")
        driver.find_element_by_id("gs_hdr_tsi").click()
        time.sleep(random.uniform(3,7))
        driver.find_element_by_id("gs_hdr_tsi").clear()
        driver.find_element_by_id("gs_hdr_tsi").send_keys("Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture")
        driver.find_element_by_id("gs_hdr_frm").submit()
        time.sleep(random.uniform(3,7))
        driver.find_element_by_xpath("(.//*[normalize-space(text()) and normalize-space(.)='Fazer login'])[1]/following::span[1]").click()
        driver.find_element_by_xpath(u"(.//*[normalize-space(text()) and normalize-space(.)='Pesquisa avançada'])[2]/following::span[2]").click()
        time.sleep(random.uniform(3,7))
        driver.find_element_by_id("gs_num-b").click()
        driver.find_element_by_link_text("20").click()
        time.sleep(random.uniform(3,7))
        driver.find_element_by_id("gs_settings_import_some").click()
        time.sleep(random.uniform(3,7))
        driver.find_element_by_name("save").click()
        time.sleep(random.uniform(3,7))
        driver.find_element_by_id("gs_hdr_tsi").click()
        time.sleep(random.uniform(3,7))
        driver.find_element_by_id("gs_hdr_tsi").clear()
        driver.find_element_by_id("gs_hdr_tsi").send_keys("Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture")
        driver.find_element_by_id("gs_hdr_frm").submit()
        time.sleep(random.uniform(3,7))

        driver.find_element_by_link_text("Importe para o BibTeX").click()
        with open("bibtex_refs.txt", "a") as f:
            f.write(driver.find_element_by_tag_name("pre").get_attribute("innerHTML") + "\n")

        driver.get("https://scholar.google.com.br/")
        time.sleep(random.uniform(3,7))
        driver.find_element_by_id("gs_hdr_tsi").send_keys("Software Platforms for Smart Cities: Concepts, Requirements, Challenges, and a Unified Reference Architecture")
        driver.find_element_by_id("gs_hdr_frm").submit()
        time.sleep(random.uniform(3,7))
        init_handle = driver.current_window_handle
        action = ActionChains(driver)

        track = {'articles': [], 'authors': []}
        titles = []
        driver.find_elements_by_partial_link_text("Citado por ")[0].click()

        try:
            for article in driver.find_elements_by_tag_name("h3"):
                titles.append(article.find_element_by_tag_name("a").get_attribute("innerHTML"))

            for article in titles:
                if article not in track['articles']:
                    track['articles'].append(article)
                    parent = driver.current_url
                    driver.get("https://scholar.google.com.br/")
                    time.sleep(random.uniform(3,7))
                    driver.find_element_by_id("gs_hdr_tsi").send_keys(article)
                    driver.find_element_by_id("gs_hdr_frm").submit()
                    time.sleep(random.uniform(3,7))
                    tex_element = driver.find_element_by_link_text("Importe para o BibTeX")
                    tex_element.click()
                    with open("bibtex_refs.txt", "a") as f:
                        f.write(driver.find_element_by_tag_name("pre").get_attribute("innerHTML") + "\n")
                    driver.get("https://scholar.google.com.br/")
                    time.sleep(random.uniform(3,7))
                    driver.find_element_by_id("gs_hdr_tsi").clear()
                    driver.find_element_by_id("gs_hdr_tsi").send_keys(article)
                time.sleep(random.uniform(3,7))
        except NoSuchElementException as e:
            driver.find_elements_by_link_text("Importe para o BibTeX").click()
            tex_element.send_keys(Keys.CONTROL + Keys.TAB)
            with open("bibtex_refs.txt", "a") as f:
                f.write(driver.find_element_by_tag_name("pre").get_attribute("innerHTML") + "\n")

        driver.close()

    def is_element_present(self, how, what):
        try: self.driver.find_element(by=how, value=what)
        except NoSuchElementException as e: return False
        return True

    def is_alert_present(self):
        try: self.driver.switch_to_alert()
        except NoAlertPresentException as e: return False
        return True

    def close_alert_and_get_its_text(self):
        try:
            alert = self.driver.switch_to_alert()
            alert_text = alert.text
            if self.accept_next_alert:
                alert.accept()
            else:
                alert.dismiss()
            return alert_text
        finally: self.accept_next_alert = True

    def tearDown(self):
        # To know more about the difference between verify and assert,
        # visit https://www.seleniumhq.org/docs/06_test_design_considerations.jsp#validating-results
        self.assertEqual([], self.verificationErrors)

if __name__ == "__main__":
    unittest.main()
