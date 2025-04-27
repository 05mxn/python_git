from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import requests
import time
import re
from datetime import datetime

# Color definitions for terminal output
class Colors:
    # Text styles
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Text colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

# Statistics for tracking downloads
class DownloadStats:
    def __init__(self):
        self.roads = {}
    
    def init_road(self, road):
        if road not in self.roads:
            self.roads[road] = {"months": {}, "total_success": 0, "total_failed": 0}
    
    def init_month(self, road, month):
        self.init_road(road)
        if month not in self.roads[road]["months"]:
            self.roads[road]["months"][month] = {"success": 0, "failed": 0}
    
    def add_success(self, road, month):
        self.init_month(road, month)
        self.roads[road]["months"][month]["success"] += 1
        self.roads[road]["total_success"] += 1
    
    def add_failure(self, road, month):
        self.init_month(road, month)
        self.roads[road]["months"][month]["failed"] += 1
        self.roads[road]["total_failed"] += 1
    
    def get_summary(self):
        summary = []
        for road in self.roads:
            road_summary = f"{road}: "
            if self.roads[road]["total_failed"] == 0:
                road_summary += "All files for all months downloaded successfully"
            else:
                road_summary += f"Downloaded {self.roads[road]['total_success']} files, {self.roads[road]['total_failed']} failed"
                # Add per-month details for failures
                failed_months = []
                for month, stats in self.roads[road]["months"].items():
                    if stats["failed"] > 0:
                        failed_months.append(f"Month {month}: {stats['failed']} files failed")
                if failed_months:
                    road_summary += " (" + ", ".join(failed_months) + ")"
            summary.append(road_summary)
        return summary

def download_file(url, file_path):
    """Helper to download a file from a URL to a local path."""
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return True

def process_download_link(link, folder_path, road, month, stats):
    """Process a download link and save the file to the specified folder."""
    link_url = link.get_attribute("href")
    if not link_url:
        stats.add_failure(road, month)
        return
        
    # Extract the text description
    link_text = link.text.strip()
    
    # Create a valid filename from the link text if it contains .zip
    if ".zip" in link_text:
        file_name = link_text
        # Remove any invalid characters from the filename
        file_name = re.sub(r'[\\/*?:"<>|]', "", file_name)
    else:
        # If there's no .zip in the text, use the last part of the URL
        file_name = link_url.split("/")[-1]
        if not file_name.endswith(".zip"):
            file_name += ".zip"
    
    file_path = os.path.join(folder_path, file_name)
    
    try:
        download_file(url=link_url, file_path=file_path)
        stats.add_success(road, month)
    except Exception as e:
        stats.add_failure(road, month)

def download_zip_files(base_url, road, year, download_dir=".", months_to_download=None, stats=None):
    """
    Use Selenium to navigate the page, click dropdowns for a specific road and year, and download .zip files.
    
    Args:
        base_url: The URL of the website
        road: Road identifier (e.g., "A57")
        year: Year to download data for
        download_dir: Base directory for downloads
        months_to_download: List of month numbers to download (1-12), or None for all months
        stats: Statistics object for tracking downloads
    """
    if stats is None:
        stats = DownloadStats()
    
    print(f"{Colors.CYAN}\n--- Processing {road} for year {year} ---{Colors.RESET}")
    
    # Create base download directory
    base_download_path = os.path.join(download_dir, road, str(year))
    os.makedirs(base_download_path, exist_ok=True)
    
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")  # Suppress most Chrome logs
    
    # Initialize the Chrome driver
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(base_url)
    
    try:
        # Step 1: Find and click the road link
        print(f"{Colors.CYAN}Looking for road: {road}{Colors.RESET}")
        # Wait for the page to fully load
        time.sleep(3)
        
        # Try different strategies to find the road
        road_element = None
        try:
            # First try: Find by exact text match
            road_xpath = f"//a[normalize-space(text())='{road}']"
            road_element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, road_xpath))
            )
        except:
            try:
                # Second try: Find by href containing the road name
                road_name = road.replace("(", "\\(").replace(")", "\\)")
                road_xpath = f"//a[contains(@href, '{road_name.lower()}')]"
                road_element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, road_xpath))
                )
            except:
                try:
                    # Third try: Find all links and look for partial text match
                    all_links = driver.find_elements(By.TAG_NAME, "a")
                    for link in all_links:
                        if road == link.text.strip():
                            road_element = link
                            break
                except:
                    pass
                
        if road_element is None:
            print(f"{Colors.RED}Could not find road: {road}{Colors.RESET}")
            return stats
        
        print(f"{Colors.CYAN}Found road link for {road}{Colors.RESET}")
        driver.execute_script("arguments[0].scrollIntoView(true);", road_element)
        driver.execute_script("arguments[0].click();", road_element)
        
        # Step 2: Wait for the road section to expand and find the year button
        time.sleep(2)
        print(f"{Colors.CYAN}Looking for year: {year}{Colors.RESET}")
        
        year_element = None
        try:
            # First try: Find by onclick attribute
            year_xpath = f"//button[contains(@onclick, \"'{road}+{year}'\")]"
            year_element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, year_xpath))
            )
        except:
            try:
                # Second try: Find by text content
                year_xpath = f"//button[contains(text(), '{year}')]"
                year_element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, year_xpath))
                )
            except:
                pass
        
        if year_element is None:
            print(f"{Colors.RED}Could not find year: {year} for road {road}{Colors.RESET}")
            return stats
        
        print(f"{Colors.CYAN}Found year button for {road} - {year}{Colors.RESET}")
        driver.execute_script("arguments[0].scrollIntoView(true);", year_element)
        driver.execute_script("arguments[0].click();", year_element)
        
        # Step 3: Wait for the content to load
        time.sleep(2)
        
        # Step 4: Find the div containing the files
        div_id = f"{road}+{year}"
        try:
            div_element = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.ID, div_id))
            )
            print(f"{Colors.CYAN}Found content div for {road} - {year}{Colors.RESET}")
        except:
            # Try alternative div ID format if roads with special characters like "(M)" don't work
            try:
                alt_road = road.replace("(", "").replace(")", "")
                alt_div_id = f"{alt_road}+{year}"
                div_element = WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.ID, alt_div_id))
                )
                print(f"{Colors.CYAN}Found content div for {road} - {year} with alternative ID{Colors.RESET}")
            except:
                print(f"{Colors.RED}Could not find content div for {road} - {year}{Colors.RESET}")
                # Take a screenshot to diagnose
                driver.save_screenshot(f"error_{road}_{year}.png")
                print(f"{Colors.YELLOW}Saved screenshot to error_{road}_{year}.png{Colors.RESET}")
                return stats
        
        # Step 5: Find all h3 elements (month indicators) and their following tables
        h3_elements = div_element.find_elements(By.TAG_NAME, "h3")
        print(f"{Colors.CYAN}Found {len(h3_elements)} months for {road} - {year}{Colors.RESET}")
        
        # Get all tables in the div for fallback method
        all_tables = div_element.find_elements(By.TAG_NAME, "table")
        
        # Track months processed
        processed_months = []
        
        # Process each month that matches our criteria
        for h3 in h3_elements:
            month_text = h3.text.strip()
            if not month_text.isdigit():
                continue  # Skip if not a numeric month
                
            month_number = int(month_text)
            
            # Skip if we're only downloading specific months and this isn't one of them
            if months_to_download is not None and month_number not in months_to_download:
                continue
                
            processed_months.append(month_number)
            
            # Find the table that follows this h3 with fallback methods
            table = None
            try:
                # First attempt: Use XPath to find table immediately following this h3
                table = h3.find_element(By.XPATH, "./following-sibling::table[1]")
            except Exception:
                try:
                    # Second attempt: Try to match by position
                    h3_index = h3_elements.index(h3)
                    if h3_index < len(all_tables):
                        table = all_tables[h3_index]
                    else:
                        # Third attempt: Look for tables that come after this h3
                        h3_location = h3.location['y']
                        potential_tables = []
                        for t in all_tables:
                            if t.location['y'] > h3_location:
                                potential_tables.append((t, t.location['y'] - h3_location))
                        if potential_tables:
                            # Get the closest table
                            table = min(potential_tables, key=lambda x: x[1])[0]
                except Exception:
                    pass
            
            if not table:
                print(f"{Colors.RED}No table found for {road} - month {month_number}{Colors.RESET}")
                continue
                
            # Create a folder for this month
            month_name = f"month_{month_number:02d}"
            month_path = os.path.join(base_download_path, month_name)
            os.makedirs(month_path, exist_ok=True)
            
            # Find download links in this table
            links = table.find_elements(By.TAG_NAME, "a")
            download_links = [link for link in links if ".zip" in link.text or "/download/" in link.get_attribute("href")]
            
            print(f"{Colors.CYAN}Processing {road} - month {month_number}: found {len(download_links)} files{Colors.RESET}")
            
            # Download files for this month
            for link in download_links:
                process_download_link(link, month_path, road, month_number, stats)
        
        if not processed_months:
            print(f"{Colors.YELLOW}No months processed for {road} - {year}{Colors.RESET}")
                
    except Exception as e:
        print(f"{Colors.RED}Error processing {road}: {e}{Colors.RESET}")
        # Take a screenshot to diagnose
        driver.save_screenshot(f"error_{road}_{year}.png")
        print(f"{Colors.YELLOW}Saved screenshot to error_{road}_{year}.png{Colors.RESET}")
    finally:
        driver.quit()
    
    return stats

def process_roads(base_url, roads, year, download_dir=".", months_to_download=None):
    """Process multiple roads and return combined statistics."""
    stats = DownloadStats()
    
    print(f"{Colors.BOLD}\n===== Starting download job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====={Colors.RESET}")
    print(f"{Colors.CYAN}Roads to process: {', '.join(roads)}{Colors.RESET}")
    print(f"{Colors.CYAN}Year: {year}{Colors.RESET}")
    print(f"{Colors.CYAN}Months: {months_to_download if months_to_download else 'All'}{Colors.RESET}")
    print(f"{Colors.CYAN}Download directory: {download_dir}{Colors.RESET}")
    print(f"{Colors.BOLD}={Colors.RESET}" * 70)
    
    for road in roads:
        download_zip_files(
            base_url=base_url, 
            road=road, 
            year=year, 
            download_dir=download_dir, 
            months_to_download=months_to_download,
            stats=stats
        )
    
    print(f"{Colors.BOLD}\n===== Download Summary ====={Colors.RESET}")
    for summary_line in stats.get_summary():
        print(summary_line)
    print(f"{Colors.BOLD}={Colors.RESET}" * 70)
    print(f"{Colors.GREEN}Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    
    return stats

if __name__ == "__main__":
    base_url = "https://tris.highwaysengland.co.uk/detail/journeytimedata"
    roads_to_download = ["M66", "M61", "M67"]
    year_to_download = "2024"
    
    # To download all months:
    months_to_download = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    process_roads(
        base_url=base_url, 
        roads=roads_to_download, 
        year=year_to_download,
        months_to_download=months_to_download
    )
