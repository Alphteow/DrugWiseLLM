"""
Author: Ilia Zenkov
Date: 9/26/2020

This script asynchronously scrapes Pubmed - an open-access database of scholarly research articles -
and saves the data to a DataFrame which is then written to a CSV intended for further processing
This script is capable of scraping a list of keywords asynchronously

Contains the following functions:
    make_header:        Makes an HTTP request header using a random user agent
    get_num_pages:      Finds number of pubmed results pages returned by a keyword search
    extract_by_article: Extracts data from a single pubmed article to a DataFrame
    get_pmids:          Gets PMIDs of all article URLs from a single page and builds URLs to pubmed articles specified by those PMIDs
    build_article_urls: Async wrapper for get_pmids, creates asyncio tasks for each page of results, page by page,
                        and stores article urls in urls: List[string]
    get_article_data:   Async wrapper for extract_by_article, creates asyncio tasks to scrape data from each article specified by urls[]

requires:
    BeautifulSoup4 (bs4)
    PANDAS
    requests
    asyncio
    aiohttp
    nest_asyncio (OPTIONAL: Solves nested async calls in jupyter notebooks)
"""

import argparse
import time
from bs4 import BeautifulSoup
import pandas as pd
import random
import requests
import ssl
import certifi
import asyncio
import aiohttp
import socket
import warnings; warnings.filterwarnings('ignore') # aiohttp produces deprecation warnings that don't concern us
#import nest_asyncio; nest_asyncio.apply() # necessary to run nested async loops in jupyter notebooks

# Use a variety of agents for our ClientSession to reduce traffic per agent
# This (attempts to) avoid a ban for high traffic from any single agent
# We should really use proxybroker or similar to ensure no ban
user_agents = [
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) Gecko/20100101 Firefox/55.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
        ]

# Create an SSL context using certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

def make_header():
    """
    Chooses a random agent from user_agents with which to construct headers.
    :return headers: dict: HTTP headers to use to get HTML from article URL.
    """
    headers = {
        'User-Agent': random.choice(user_agents),
    }
    return headers

async def extract_by_article(url):
    """
    Extracts all data from a single article.
    :param url: string: URL to a single article (i.e. root PubMed URL + PMID).
    :return article_data: Dict: Contains all data from a single article.
    """
    conn = aiohttp.TCPConnector(family=socket.AF_INET, ssl=ssl_context)  # Use SSL context here
    headers = make_header()
    global articles_data
    async with aiohttp.ClientSession(headers=headers, connector=conn) as session:
        async with semaphore, session.get(url) as response:
            data = await response.text()
            soup = BeautifulSoup(data, "lxml")
            # Extract article data
            try:
                abstract_raw = soup.find('div', {'class': 'abstract-content selected'}).find_all('p')
                abstract = ' '.join([paragraph.text.strip() for paragraph in abstract_raw])
            except:
                abstract = 'NO_ABSTRACT'
            try:
                affiliations = [affiliation.get_text().strip() for affiliation in soup.find('ul', {'class': 'item-list'}).find_all('li')]
            except:
                affiliations = 'NO_AFFILIATIONS'
            try:
                keywords = soup.find('div', {'class': 'abstract'}).find_all('p')[-1].get_text().replace('Keywords:', '').strip()
            except:
                keywords = 'NO_KEYWORDS'
            try:
                title = soup.find('meta', {'name': 'citation_title'})['content'].strip('[]')
            except:
                title = 'NO_TITLE'
            try:
                authors = ', '.join([author.text for author in soup.find('div', {'class': 'authors-list'}).find_all('a', {'class': 'full-name'})])
            except:
                authors = 'NO_AUTHOR'
            try:
                journal = soup.find('meta', {'name': 'citation_journal_title'})['content']
            except:
                journal = 'NO_JOURNAL'
            try:
                date = soup.find('time', {'class': 'citation-year'}).text
            except:
                date = 'NO_DATE'

            article_data = {
                'url': url,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'affiliations': affiliations,
                'journal': journal,
                'keywords': keywords,
                'date': date
            }
            articles_data.append(article_data)

async def get_pmids(page, keyword):
    """
    Extracts PMIDs of all articles from a PubMed search result, page by page,
    builds a URL to each article, and stores all article URLs in urls: List[string].
    :param page: int: Value of current page of a search result for keyword.
    :param keyword: string: Current search keyword.
    :return: None
    """
    page_url = f'{pubmed_url}+{keyword}+&page={page}'
    headers = make_header()
    conn = aiohttp.TCPConnector(family=socket.AF_INET, ssl=ssl_context)  # Use SSL context here
    async with aiohttp.ClientSession(headers=headers, connector=conn) as session:
        async with session.get(page_url) as response:
            data = await response.text()
            soup = BeautifulSoup(data, "lxml")
            pmids = soup.find('meta', {'name': 'log_displayeduids'})['content']
            for pmid in pmids.split(','):
                url = root_pubmed_url + '/' + pmid
                urls.append(url)

def get_num_pages(keyword):
    '''
    Gets total number of pages returned by search results for keyword
    :param keyword: string: search word used to search for results
    :return: num_pages: int: number of pages returned by search results for keyword
    '''
    # Return user specified number of pages if option was supplied
    if args.pages != None: 
        return args.pages

    # Get search result page and wait a second for it to load
    # URL to the first page of results for a keyword search
    headers = make_header()
    search_url = f'{pubmed_url}+{keyword}'
    with requests.get(search_url, headers=headers) as response:
        data = response.text
        soup = BeautifulSoup(data, "lxml")
        num_pages = int((soup.find('span', {'class': 'total-pages'}).get_text()).replace(',', ''))
        return num_pages  # Can hardcode this value (e.g. 10 pages) to limit # of articles scraped per keyword
    

async def build_article_urls(keywords):
    """
    Async wrapper for get_pmids, page by page of results, for a single search keyword.
    Creates an asyncio task for each page of search result for each keyword.
    :param keywords: List[string]: List of search keywords.
    :return: None
    """
    tasks = []
    for keyword in keywords:
        num_pages = get_num_pages(keyword)
        for page in range(1, num_pages + 1):
            task = asyncio.create_task(get_pmids(page, keyword))
            tasks.append(task)

    await asyncio.gather(*tasks)

async def get_article_data(urls):
    """
    Async wrapper for extract_by_article to scrape data from each article (url).
    :param urls: List[string]: List of all PubMed URLs returned by the search keyword.
    :return: None
    """
    tasks = []
    for url in urls:
        if url not in scraped_urls:
            task = asyncio.create_task(extract_by_article(url))
            tasks.append(task)
            scraped_urls.append(url)

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    print("Starting...")
    parser = argparse.ArgumentParser(description='Asynchronous PubMed Scraper')
    parser.add_argument('--pages', type=int, default=None, help='Specify number of pages to scrape for EACH keyword.')
    parser.add_argument('--start', type=int, default=2019, help='Specify start year for publication date range to scrape.')
    parser.add_argument('--stop', type=int, default=2020, help='Specify stop year for publication date range to scrape.')
    parser.add_argument('--output', type=str, default='articles.csv', help='Choose output file name.')
    args = parser.parse_args()
    if args.output[-4:] != '.csv': args.output += '.csv'

    start = time.time()
    pubmed_url = f'https://pubmed.ncbi.nlm.nih.gov/?term={args.start}%3A{args.stop}%5Bdp%5D'
    root_pubmed_url = 'https://pubmed.ncbi.nlm.nih.gov'
    search_keywords = []
    with open('keywords.txt') as file:
        keywords = file.readlines()
        [search_keywords.append(keyword.strip()) for keyword in keywords]

    articles_data = []
    urls = []
    scraped_urls = []
    semaphore = asyncio.BoundedSemaphore(100)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(build_article_urls(search_keywords))
    print(f'Scraping initiated for {len(urls)} article URLs found from {args.start} to {args.stop}\n')
    loop.run_until_complete(get_article_data(urls))

    articles_df = pd.DataFrame(articles_data, columns=['title', 'abstract', 'affiliations', 'authors', 'journal', 'date', 'keywords', 'url'])
    print('Preview of scraped article data:\n')
    print(articles_df.head(5))
    filename = args.output
    articles_df.to_csv(filename)
    print(f'It took {time.time() - start} seconds to find {len(urls)} articles; {len(scraped_urls)} unique articles were saved to {filename}')

