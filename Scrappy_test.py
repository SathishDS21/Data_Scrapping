import pandas as pd
import requests
from bs4 import BeautifulSoup
def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        urls = df['Links'].tolist()
        return urls
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []
def scrape_urls(url_list):
    results = []
    for url in url_list:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                title = soup.title.string if soup.title else 'No title found'

                meta_description = soup.find('meta', attrs={'name': 'description'})
                summary = ''

                if meta_description and 'content' in meta_description.attrs:
                    summary = meta_description['content']
                else:

                    first_paragraph = soup.find('p')
                    summary = first_paragraph.get_text() if first_paragraph else 'No summary found'

                print(f"Title of {url}: {title}")
                print(f"Summary of {url}: {summary}")
                results.append({'URL': url, 'Title': title, 'Summary': summary})
            else:
                print(f"Failed to retrieve {url}, status code: {response.status_code}")
                results.append({'URL': url, 'Title': f"Failed (Status code: {response.status_code})", 'Summary': ''})
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            results.append({'URL': url, 'Title': f"Error: {e}", 'Summary': ''})
    return results
def save_to_excel(data, output_file_path):
    try:
        df = pd.DataFrame(data)
        df.to_excel(output_file_path, index=False)
        print(f"Scraping results saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        url = 'https://www.firstpost.com/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
if __name__ == "__main__":
    excel_file_path = "/Users/sathishm/Documents/TSM Folder/Scrapping Input data/Test_data_first.xlsx"
    output_file_path = "/Users/sathishm/Documents/TSM Folder/Scrapping Output data/Test_data_output.xlsx"

    urls_to_scrape = read_excel(excel_file_path)
    scraped_data = scrape_urls(urls_to_scrape)
    save_to_excel(scraped_data, output_file_path)