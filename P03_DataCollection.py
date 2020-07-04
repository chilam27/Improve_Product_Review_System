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

customer_id = []
customer_name = []
rating = []
review_date = []
review_loc = []
size = []
color = []
verified_purchase = []
review_header = []
review_body = []

html1 = 'https://www.amazon.com/Dickies-Mens-Original-874-Work/product-reviews/B07PFLQ73D/ref=cm_cr_arp_d_paging_btm_'
html2 = '?ie=UTF8&pageNumber='
html3 = '&reviewerType=all_reviews'
page = 1


for count in soup.find_all('span', attrs = {'data-hook' : 'cr-filter-info-review-count'}):
    count = count.text.split(' ')[-2]
    total_page = round(int(count.replace(',','')) / 10, 0) # find total of review pages the product has
 

while (page < (total_page + 1)):
    for review in soup.find_all('div', attrs = {'class' : 'a-section a-spacing-none review-views celwidget'}):
        for cus_ids in review.find_all('div', attrs = {'class' : 'a-section review aok-relative'}):
            for cus_id in cus_ids:
                cus_id = cus_id.attrs['id'].split('-')[0]
                customer_id.append(cus_id) # append 'customer_id'
        
        for cus_name in review.find_all('span', attrs = {'class' : 'a-profile-name'}):
            customer_name.append(cus_name.text) # append 'customer_name'
            
        for rate in review.find_all('span', attrs = {'class' : 'a-icon-alt'}):
            rate = rate.text.split(' ')[0]
            rating.append(rate) #append 'rating'
            
        for date in review.find_all('span', attrs = {'class' : 'a-size-base a-color-secondary review-date'}):
            date = date.text
            location = ' '.join(date.split()[2:-4])
            date = ' '.join(date.split()[-3:])
            date = date.split(' ')[0] + ' ' + date.split(' ')[-1]
            review_loc.append(location)
            review_date.append(date) # append 'review_date' and 'review_loc'
            
        for siz in review.find_all('a', attrs = {'class' : 'a-size-mini a-link-normal a-color-secondary'}):
            siz = siz.text.replace('Color:', '')
            colo = ' '.join(siz.split()[4:])
            siz = ' '.join(siz.split()[1:4])
            color.append(colo)
            size.append(siz) # append 'size' and 'color'
            
        for verify in review.find_all('span', attrs = {'class' : 'a-size-mini a-color-state a-text-bold'}):
            if not verify:
                verify = "na"
            verified_purchase.append(verify.text) # append 'verified_purchase'
            
        for head in review.find_all('a', attrs = {'data-hook' : 'review-title'}):
            review_header.append(head.text) # append 'review_header'
            
        for body in review.find_all('span', attrs = {'class' : 'a-size-base review-text review-text-content'}):
            review_body.append(body.text) # append 'review_body'
    
    page = page + 1
    page_num = str(page)
    html = html1 + page_num + html2 + page_num + html3 # switch to the next review page
    
    result = requests.get(html, headers= get_headers())
    soup = BeautifulSoup(result.content, 'lxml')

