# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:45:31 2020

@author: Chi Lam
"""

import requests
from bs4 import BeautifulSoup


def get_headers():
    headers = {'accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-language':'en-US,en;q=0.9',
            'cache-control':'max-age=0',
            'upgrade-insecure-requests':'1',
            'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'}
    return headers


result = requests.get("https://www.amazon.com/Dickies-Mens-Original-874-Work/product-reviews/B07PFLQ73D/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews", headers= get_headers())
soup = BeautifulSoup(result.content, 'lxml')
print(soup)

customer_id = []
customer_name = []
rating = []
review_date = []
size = []
color = []
verified_purchase = []
review_header = []
review_body = []


for review in soup.find_all('div', attrs = {'class' : 'a-section a-spacing-none review-views celwidget'}):
    for cus_ids in review.find_all('div', attrs = {'class' : 'a-section review aok-relative'}):
        for cus_id in cus_ids:
            customer_id.append(cus_id.attrs['id']) # append 'customer_id'
    
    for cus_name in review.find_all('span', attrs = {'class' : 'a-profile-name'}):
        customer_name.append(cus_name.text) # append 'customer_name'
        
    
    



