from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import tempfile
import json


def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"--user-data-dir={tempfile.mkdtemp()}")
    return webdriver.Chrome(options=options)


def get_soup(url):
    driver = setup_driver()
    driver.get(url)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    return soup


def get_sectors():
    url = "https://www.ncs.gov.in/content-repository/Pages/BrowseBySectors.aspx"
    print(f"[DEBUG] Fetching sectors from: {url}")
    soup = get_soup(url)
    sectors = []
    for div in soup.select("div.col-sm-6.col-md-3.item.marginTop30"):
        a_tag = div.find("a", href=True)
        if a_tag:
            name = a_tag.get_text(strip=True)
            link = urljoin(url, a_tag["href"])
            print(f"[DEBUG] Found sector: {name} — {link}")
            sectors.append((name, link))
    return sectors


def get_jobs_for_sector(sector_url):
    print(f"[DEBUG] Getting jobs from: {sector_url}")
    soup = get_soup(sector_url)
    jobs = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "ViewNcoDetails.aspx?NCSCode=" in href:
            full_url = urljoin(sector_url, href)
            code = full_url.split("NCSCode=")[-1]
            if code not in seen:
                title = a.get_text(strip=True)
                seen.add(code)
                jobs.append((title, full_url))
    return jobs


def parse_job_details(job_url):
    print(f"[DEBUG] Parsing job details from: {job_url}")
    soup = get_soup(job_url)
    details = {}

    h1 = soup.find("h1")
    details["Title"] = h1.get_text(strip=True) if h1 else ""

    def extract_section(title):
        headers = soup.find_all("div", class_="NCOpanel-heading")
        for header in headers:
            if header.get_text(strip=True).startswith(title):
                body = header.find_next_sibling("div", class_="NCOpanel-body")
                if body:
                    items = body.find_all("li")
                    if items:
                        return "; ".join(li.get_text(" ", strip=True) for li in items)
                    return body.get_text(" ", strip=True)
        return ""

    details["Job Description"] = extract_section("JOB DESCRIPTION").replace("\n", " ").strip()
    details["Educational Qualifications And Training"] = extract_section("EDUCATIONAL QUALIFICATIONS AND TRAINING").replace("\n", " ").strip()
    details["Key Competencies"] = extract_section("KEY COMPETENCIES").replace("\n", " ").strip()

    return details


def main():
    sectors = get_sectors()
    if not sectors:
        print("No sectors found.")
        return

    print(f"\n[INFO] Scraping only the first sector and its first 3 jobs...\n")

    all_jobs = []

    sector_name, sector_url = sectors[0]  # Only the first sector
    print(f"\n[INFO] Sector: {sector_name}")
    try:
        jobs = get_jobs_for_sector(sector_url)
        for title, job_url in jobs:  # First 3 jobs only
            details = parse_job_details(job_url)
            job_data = {
                "Sector": sector_name,
                "Title": details["Title"],
                "Job Description": details["Job Description"],
                "Educational Qualifications And Training": details["Educational Qualifications And Training"],
                "Key Competencies": details["Key Competencies"],
            }
            all_jobs.append(job_data)
            print(f"  ✔ {job_data['Title']}")
    except Exception as e:
        print(f"[!] Error in {sector_name}: {e}")

    with open("ncs_sample_jobs.json", "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, indent=2, ensure_ascii=False)

    print("\n✅ Done! Data saved to `ncs_sample_jobs.json`\n")


if __name__ == "__main__":
    main()
