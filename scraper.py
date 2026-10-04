import os
import json
import time
import re
import hashlib
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
    WebDriverException,
)
from webdriver_manager.chrome import ChromeDriverManager

# Set HEADLESS=false so you can see the browser window and solve the CAPTCHA manually.
HEADLESS = os.getenv("HEADLESS", "false").lower() != "false"

# --- Hard limits so a bad page/network/CAPTCHA can never hang the script forever ---
PAGE_LOAD_TIMEOUT = 30          # seconds - driver.get() / navigation
SCRIPT_TIMEOUT = 20             # seconds - any driver.execute_script call
CAPTCHA_WAIT_TIMEOUT = 180      # seconds - how long we'll wait for a human to solve a CAPTCHA
SCROLL_WALL_CLOCK_BUDGET = 180  # seconds - hard cap on total time spent scrolling one dialog
COMPETITOR_WALL_CLOCK_BUDGET = 900 # seconds - hard cap per competitor before we give up and move on

# Any of these appearing on the page means Google wants a human to verify.
CAPTCHA_MARKERS = [
    "unusual traffic",
    "verify you're a human",
    "recaptcha",
    "our systems have detected unusual",
]

# Buttons that end a single Google Maps "update" card - used to split raw text into posts.
CTA_LABELS = ["Book", "Call now", "Learn more", "Order online", "Sign up", "Buy", "Get offer"]


class CaptchaBlocked(Exception):
    """Raised when Google shows a verification screen and no one solved it in time."""
    pass


class NoUpdatesFound(Exception):
    """The business panel loaded fine, but the place has never posted Google updates."""
    pass


class ScrapeTimeout(Exception):
    """Raised when a single operation blew through its wall-clock budget."""
    pass


def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    if HEADLESS:
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")
    else:
        options.add_argument("--start-maximized")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    # Without these, a slow/hung network request or a runaway execute_script can
    # block the whole process indefinitely - this is the #1 cause of a scraper
    # that "just hangs" with no error.
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.set_script_timeout(SCRIPT_TIMEOUT)
    return driver


def _timed_input(prompt: str, timeout: int) -> Optional[str]:
    """
    input() that gives up after `timeout` seconds instead of blocking forever.
    Needed because plain input() will hang the whole process indefinitely if
    the script ever runs unattended (cron, server, CI) and nobody is there to
    press Enter.
    """
    result_q: "queue.Queue[str]" = queue.Queue()

    def _read():
        try:
            result_q.put(input(prompt))
        except Exception:
            result_q.put("")

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    try:
        return result_q.get(timeout=timeout)
    except queue.Empty:
        return None


def check_and_solve_captcha(driver) -> bool:
    """
    Human-in-the-Loop (HITL) CAPTCHA handler.
    Detects if a CAPTCHA or block screen is present, pauses the script,
    and waits (up to CAPTCHA_WAIT_TIMEOUT seconds) for the user to manually
    solve it in the browser. If nobody responds in time, it gives up loudly
    (raises CaptchaBlocked) instead of hanging the process forever - the
    caller decides whether to skip this competitor or abort the run.
    """
    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text.lower()
        is_blocked = any(marker in body_text for marker in CAPTCHA_MARKERS)

        # Also check if a reCAPTCHA iframe is present on the page
        if not is_blocked:
            if driver.find_elements(By.XPATH, '//iframe[contains(@title, "reCAPTCHA")]'):
                is_blocked = True

        if is_blocked:
            print("\n" + "=" * 70)
            print("[!] CAPTCHA / BOT DETECTION TRIGGERED!")
            print(f"[!] Solve it in the browser within {CAPTCHA_WAIT_TIMEOUT}s, then press [ENTER] here.")
            print("[!] (Ensure HEADLESS=false is set if you cannot see the browser window).")
            print("=" * 70 + "\n")

            response = _timed_input(
                ">>> Press [ENTER] AFTER you have solved the CAPTCHA (or wait to auto-skip)... ",
                CAPTCHA_WAIT_TIMEOUT,
            )

            if response is None:
                print(f"[!] No response after {CAPTCHA_WAIT_TIMEOUT}s - giving up on this page.")
                raise CaptchaBlocked("CAPTCHA not solved within timeout window")

            time.sleep(3)
            print("[+] Resuming script execution...")
            return True

    except CaptchaBlocked:
        raise
    except Exception:
        pass

    return False


def page_is_captcha(driver) -> bool:
    # Note: raises CaptchaBlocked if a CAPTCHA appears and nobody solves it in
    # time - callers should catch that rather than let it silently hang.
    return check_and_solve_captcha(driver)


def dismiss_consent_dialog(driver):
    """
    Google shows a 'Before you continue to Google Maps' cookie interstitial on
    fresh/cookie-less sessions - very common. If we don't click through it,
    the actual page never loads properly.
    """
    consent_xpaths = [
        '//button[contains(., "Accept all")]',
        '//button[contains(., "I agree")]',
        '//button[contains(., "Reject all")]',
        '//form[contains(@action, "consent")]//button',
    ]
    for xpath in consent_xpaths:
        buttons = driver.find_elements(By.XPATH, xpath)
        if buttons:
            try:
                buttons[0].click()
                time.sleep(2)
                return True
            except Exception:
                continue
    return False


PLACE_ID_PATTERN = re.compile(r"!1s(0x[0-9a-f]+:0x[0-9a-f]+)")


def extract_place_id(maps_url: str) -> Optional[str]:
    match = PLACE_ID_PATTERN.search(maps_url)
    return match.group(1) if match else None

def scroll_sidebar_to_load_all(driver, max_scrolls=20):
    """
    Scrolls down the Google Maps left feed panel iteratively 
    to trigger the loading of more search results.
    """
    print("[*] Scrolling sidebar to load more results...")
    try:
        # The main results container feed XPath
        scrollable_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//div[@role="feed"]'))
        )
    except Exception:
        print("[-] Feed container not found, skipping scroll.")
        return

    last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
    
    for i in range(max_scrolls):
        # Scroll down to the bottom of the feed container
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
        time.sleep(2)  # Wait for new elements to render via AJAX
        
        # Check if new content has loaded by measuring new height
        new_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)
        if new_height == last_height:
            # Reached the end of the results list
            print(f"[*] Reached the end of the results list after {i+1} scrolls.")
            break
        last_height = new_height

_NOT_AN_ADDRESS = re.compile(
    r"^(open|closed|closes|opens|temporarily|permanently|dine-in|takeaway|take-away|"
    r"delivery|no delivery|no takeaway|no dine-in|curbside|drive-through|"
    r"\d(\.\d)?\s*\(|\d(\.\d)?$|\u20b9|rs\.?\s*\d|\$)",
    re.IGNORECASE,
)


def _address_from_card_text(card_text: str, name: str = "") -> Optional[str]:
    """
    A Maps result card reads like:
        Jimmy's Burger
        4.2(120) · ₹200–400
        Fast food restaurant · Shop 5, Sector 17, Vashi
        Open ⋅ Closes 11 pm
    The address is the part after the category on the 3rd line.
    """
    for raw in (card_text or "").splitlines():
        line = raw.strip()
        if not line or line.casefold() == (name or "").strip().casefold():
            continue
        parts = [p.strip() for p in re.split(r"[\u00b7\u22c5\u2022]", line) if p.strip()]
        if len(parts) < 2:
            continue
        candidate = parts[-1]
        if _NOT_AN_ADDRESS.match(candidate) or _NOT_AN_ADDRESS.match(parts[0]):
            continue
        if len(candidate) < 4:
            continue
        return candidate
    return None


def _read_place_address(driver) -> Optional[str]:
    """Address from an opened Google Maps place page."""
    selectors = [
        (By.CSS_SELECTOR, "button[data-item-id='address']"),
        (By.XPATH, "//button[starts-with(@aria-label, 'Address:')]"),
    ]
    for by, selector in selectors:
        try:
            for el in driver.find_elements(by, selector):
                label = (el.get_attribute("aria-label") or el.text or "").strip()
                label = re.sub(r"^\s*Address:\s*", "", label, flags=re.IGNORECASE).strip()
                if label:
                    return label
        except Exception:
            continue
    return None


def build_search_query(name: str, address: Optional[str] = None) -> str:
    """
    'Jimmy's Burger' alone matches every branch. Adding the branch's address
    makes Google open the knowledge panel of that exact place.
    """
    name = " ".join((name or "").split())
    address = " ".join((address or "").split())
    if not address:
        return name
    if address.casefold().startswith(name.casefold()):
        return address[:160]
    return f"{name} {address}"[:200]


def search_place_candidates(query: str) -> List[Dict[str, Any]]:
    driver = init_driver()
    try:
        search_url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"
        try:
            driver.get(search_url)
        except TimeoutException:
            print(f"[-] Page load timed out after {PAGE_LOAD_TIMEOUT}s for query: {query}")
            return []
        time.sleep(5)

        dismiss_consent_dialog(driver)

        try:
            page_is_captcha(driver)
        except CaptchaBlocked:
            print(f"[-] CAPTCHA not solved in time for query: {query}")
            return []

        if "/maps/place/" in driver.current_url:
            name = None
            try:
                name = driver.find_element(By.TAG_NAME, "h1").text.strip()
            except Exception:
                pass
            if not name:
                name = driver.title.replace(" - Google Maps", "").strip()

            current_url = driver.current_url
            return [{
                "name": name or query,
                "address": _read_place_address(driver),
                "maps_url": current_url,
                "place_id": extract_place_id(current_url),
            }]

        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, '//a[contains(@href, "/maps/place/")]'))
            )
        except Exception:
            return []
        scroll_sidebar_to_load_all(driver, max_scrolls=15)
        result_links = driver.find_elements(By.XPATH, '//div[@role="feed"]//a[contains(@href, "/maps/place/")]')
        if not result_links:
            result_links = driver.find_elements(By.XPATH, '//a[contains(@href, "/maps/place/")]')

        candidates = []
        seen_urls = set()
        for link in result_links:
            href = link.get_attribute("href")
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)

            name = link.get_attribute("aria-label") or link.text.strip()
            if not name:
                continue

            address = None
            try:
                card = link.find_element(By.XPATH, "./ancestor::div[@role='article'][1]")
                address = _address_from_card_text(card.text, name)
            except Exception:
                pass

            candidates.append({
                "name": name,
                "address": address,
                "maps_url": href,
                "place_id": extract_place_id(href),
            })

        return candidates

    finally:
        driver.quit()


# =====================================================================
# POST SCRAPER  (ported from test7.py - DOM/attribute based)
#
# Everything below this line replaces the old test4-derived post code.
# It does not rely on Google's generated CSS classes; it keys off stable
# attributes (data-post-id, data-media-index, role/aria-level, jsname).
# =====================================================================

GOOGLE_URL = "https://www.google.com/?hl=en"   # hl=en: the selectors match English UI text

DEFAULT_WAIT = 15
SCROLL_PAUSE = 1.2
MAX_SCROLL_ROUNDS = 40
MAX_DUPLICATE_HITS = 3  # stop a scrape once 3 repeated post-content fingerprints are seen
# Re-scrape shortcut: Google lists posts newest-first, so once this many posts IN A ROW
# are already saved in the database, everything older is saved too -> stop, we're up to date.
MAX_KNOWN_STREAK = 3
NO_NEW_POST_ROUNDS = 4


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


@dataclass
class GooglePost:
    restaurant: Optional[str] = None
    date: Optional[str] = None
    title: Optional[str] = None
    validity: Optional[str] = None
    content: Optional[str] = None

    image_url: Optional[str] = None
    video_url: Optional[str] = None

    post_id: Optional[str] = None
    feature_id: Optional[str] = None
    content_id: Optional[str] = None

    post_url: Optional[str] = None
    cta_text: Optional[str] = None


class GoogleUpdatesScraper:
    """
    Two-phase scraper (same architecture as test7.py).

    PHASE 1  scroll the Updates popup and read every field that is already
             present in each card. No Share clicks happen here.
    PHASE 2  one post at a time, restore the popup if needed, click that
             card's own Share button and read the share.google URL.

    It works on an already-created Selenium driver, so run_competitor_scrape
    keeps control of the browser lifecycle and the watchdog.
    """

    def __init__(
        self,
        driver,
        competitor_name: str = "",
        is_known: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        self.driver = driver
        self.competitor_name = competitor_name
        # Callback supplied by main.py: takes the post dict (same shape that
        # run_competitor_scrape returns) and says whether it is already in the DB.
        self.is_known = is_known
        self.known_streak = 0       # consecutive already-saved posts (resets on a new post)
        self.known_total = 0        # already-saved posts seen this run
        self.posts_checked = 0      # posts read this run (new + already saved)
        self.up_to_date = False     # True when we stopped because of MAX_KNOWN_STREAK
        self.wait = WebDriverWait(driver, DEFAULT_WAIT)
        # post_id -> GooglePost. Kept on the instance so that if the run is
        # cut short (watchdog / CAPTCHA) the caller can still keep what was
        # collected so far.
        self.collected: Dict[str, GooglePost] = {}
        self.seen_post_ids: set[str] = set()
        # Google can expose the same visible post under different IDs.
        # Fingerprints stop those content-level duplicates.
        self.collected_fingerprints: set[str] = set()
        self.duplicate_hits = 0
        self.stop_requested = False

    # ------------------------------------------------------------------
    # basic helpers
    # ------------------------------------------------------------------

    def sleep(self, seconds: float):
        time.sleep(seconds)

    def visible(self, element) -> bool:
        try:
            return element.is_displayed()
        except Exception:
            return False

    def safe_text(self, element) -> str:
        try:
            return element.text.strip()
        except Exception:
            return ""

    def safe_attribute(self, element, attribute: str) -> Optional[str]:
        try:
            value = element.get_attribute(attribute)
            if value:
                return value.strip()
        except Exception:
            pass
        return None

    def click_element(self, element) -> bool:
        """Normal Selenium click first, JavaScript click as fallback."""
        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block:'center', inline:'center'});",
                element,
            )
            self.sleep(0.3)
            element.click()
            return True
        except (ElementClickInterceptedException, WebDriverException):
            pass

        try:
            self.driver.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # CAPTCHA (polls until a human solves it, but never forever)
    # ------------------------------------------------------------------

    def captcha_detected(self) -> bool:
        captcha_selectors = [
            "iframe[src*='recaptcha']",
            "iframe[title*='reCAPTCHA']",
            "#captcha-form",
            "form[action*='Captcha']",
            "form[action*='captcha']",
        ]

        for selector in captcha_selectors:
            try:
                for element in self.driver.find_elements(By.CSS_SELECTOR, selector):
                    if self.visible(element):
                        return True
            except Exception:
                continue

        page_text = ""
        try:
            page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
        except Exception:
            pass

        captcha_words = [
            "unusual traffic",
            "not a robot",
            "verify you're human",
            "verify you are human",
            "captcha",
        ]
        return any(word in page_text for word in captcha_words)

    def wait_for_captcha_if_needed(self):
        if not self.captcha_detected():
            return

        print("\n" + "=" * 70)
        print("[!] CAPTCHA DETECTED - solve it manually in the Chrome window.")
        print(f"[!] Waiting up to {CAPTCHA_WAIT_TIMEOUT}s, then continuing automatically.")
        print("=" * 70)

        deadline = time.monotonic() + CAPTCHA_WAIT_TIMEOUT
        while self.captcha_detected():
            if time.monotonic() > deadline:
                raise CaptchaBlocked("CAPTCHA not solved within timeout window")
            self.sleep(2)

        print("[+] CAPTCHA solved. Continuing...")
        self.sleep(2)

    # ------------------------------------------------------------------
    # Google search
    # ------------------------------------------------------------------

    def search(self, query: str):
        print(f"[*] Searching Google for: {query}")

        self.driver.get(GOOGLE_URL)
        self.sleep(2)

        dismiss_consent_dialog(self.driver)
        self.wait_for_captcha_if_needed()

        search_box = None
        selectors = [
            (By.NAME, "q"),
            (By.CSS_SELECTOR, "textarea[name='q']"),
            (By.CSS_SELECTOR, "input[name='q']"),
        ]

        for by, selector in selectors:
            try:
                search_box = self.wait.until(EC.presence_of_element_located((by, selector)))
                if self.visible(search_box):
                    break
            except TimeoutException:
                continue

        if search_box is None:
            raise RuntimeError("Google search box could not be found.")

        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.ENTER)

        self.sleep(3)
        self.wait_for_captcha_if_needed()
        print("[+] Google search completed.")

    # ------------------------------------------------------------------
    # "View previous updates on Google" trigger
    # ------------------------------------------------------------------

    def find_updates_trigger(self):
        selectors = [
            (By.XPATH, "//*[normalize-space()='View previous updates on Google']"),
            (By.XPATH, "//*[contains(normalize-space(), 'View previous updates on Google')]"),
        ]

        for by, selector in selectors:
            try:
                for element in self.driver.find_elements(by, selector):
                    if self.visible(element):
                        return element
            except Exception:
                continue

        # Context-based fallback: look next to the "Updates from ..." heading.
        headings = self.driver.find_elements(
            By.XPATH, "//div[@role='heading' and @aria-level='3']"
        )

        for heading in headings:
            if "Updates from" not in self.safe_text(heading):
                continue

            for xpath in [
                ".//*[contains(normalize-space(), 'View previous updates')]",
                "./following::*[contains(normalize-space(), 'View previous updates')][1]",
            ]:
                try:
                    for element in heading.find_elements(By.XPATH, xpath):
                        if self.visible(element):
                            return element
                except Exception:
                    continue

        return None

    def business_panel_present(self) -> bool:
        """True if Google opened a business knowledge panel (Directions / reviews buttons)."""
        xpaths = [
            "//*[self::a or self::button or @role='button'][normalize-space()='Directions' or .//*[normalize-space()='Directions']]",
            "//*[normalize-space()='Write a review']",
            "//*[normalize-space()='Google reviews']",
            "//a[contains(@href,'/maps/dir/')]",
        ]
        for xp in xpaths:
            try:
                for el in self.driver.find_elements(By.XPATH, xp):
                    if self.visible(el):
                        return True
            except Exception:
                continue
        return False

    def open_updates(self):
        print("[*] Looking for the Updates section...")

        trigger = None
        for _ in range(10):
            trigger = self.find_updates_trigger()
            if trigger is not None:
                break
            self.sleep(1)

        if trigger is None:
            if self.business_panel_present():
                raise NoUpdatesFound(
                    f"{self.competitor_name or 'This place'} has not posted anything "
                    "until now on google updates"
                )
            raise RuntimeError(
                "Could not find the business on Google (no business panel opened). "
                "Check the name/address and try again."
            )

        print("[+] Found: View previous updates on Google")

        if not self.click_element(trigger):
            raise RuntimeError("Could not click 'View previous updates on Google'.")

        print("[+] Clicked Updates trigger.")

        if not self.wait_for_updates_popup():
            raise RuntimeError("Updates popup did not open.")

        print("[+] Updates popup detected.")

    # ------------------------------------------------------------------
    # Updates popup
    # ------------------------------------------------------------------

    def find_updates_popup(self):
        """Google exposes the popup title as role=heading aria-level=2 'Updates'."""
        selectors = [
            (
                By.XPATH,
                "//div[@role='heading' and @aria-level='2' and normalize-space()='Updates']",
            ),
            (By.XPATH, "//*[self::h2 or self::div][normalize-space()='Updates']"),
        ]

        for by, selector in selectors:
            try:
                for element in self.driver.find_elements(by, selector):
                    if self.visible(element):
                        return element
            except Exception:
                continue

        return None

    def wait_for_updates_popup(self) -> bool:
        for _ in range(DEFAULT_WAIT * 2):
            if self.find_updates_popup() is not None:
                return True
            self.sleep(0.5)
        return False

    def ensure_updates_popup_open(self) -> bool:
        """Share can close the popup, so Phase 2 re-checks before every post."""
        if self.find_updates_popup() is not None:
            return True

        print("    Updates popup is closed. Reopening...")
        try:
            self.open_updates()
            return self.find_updates_popup() is not None
        except Exception as exc:
            print(f"    Could not reopen Updates popup: {exc}")
            return False

    # ------------------------------------------------------------------
    # Post cards
    # ------------------------------------------------------------------

    def get_posts(self):
        """
        Currently rendered cards. Google can swap the DOM mid-render, so
        several independent lookups are tried plus a JS fallback.
        """
        selectors = [
            (By.CSS_SELECTOR, "article[data-post-id]"),
            (By.XPATH, "//article[@data-post-id]"),
            (By.XPATH, "//*[@data-post-id and self::article]"),
        ]

        for by, selector in selectors:
            try:
                elements = self.driver.find_elements(by, selector)
                if elements:
                    return elements
            except Exception:
                continue

        try:
            elements = self.driver.execute_script(
                "return Array.from(document.querySelectorAll('article[data-post-id]'));"
            )
            if elements:
                return elements
        except Exception:
            pass

        return []

    def wait_for_post_cards(self, timeout=12):
        """Wait for Google to actually render article[data-post-id] cards."""
        deadline = time.time() + timeout

        while time.time() < deadline:
            posts = self.get_posts()
            if posts:
                return posts

            try:
                self.driver.execute_script("return document.body.offsetHeight;")
            except Exception:
                pass

            self.sleep(0.4)

        return self.get_posts()

    def refind_post(self, post_id: str, attempts: int = 10):
        """Fresh article element by its stable data-post-id."""
        selector = f"//article[@data-post-id='{post_id}']"

        for _ in range(attempts):
            try:
                for article in self.driver.find_elements(By.XPATH, selector):
                    if self.visible(article):
                        return article
            except Exception:
                pass
            self.sleep(0.25)

        return None

    # ------------------------------------------------------------------
    # Scroll container
    # ------------------------------------------------------------------

    def find_scroll_container(self):
        """
        Find the popup's scrollable element WITHOUT needing a card to exist
        (Google lazy-renders the cards).
        """
        try:
            popup_heading = self.find_updates_popup()
            if popup_heading is None:
                return None

            return self.driver.execute_script(
                """
                const heading = arguments[0];

                // 1) ancestors of the Updates heading
                let el = heading;
                while (el) {
                    const style = window.getComputedStyle(el);
                    const overflowY = style.overflowY;
                    if ((overflowY === 'auto' || overflowY === 'scroll') &&
                        el.scrollHeight > el.clientHeight + 5) {
                        return el;
                    }
                    el = el.parentElement;
                }

                // 2) descendants of the nearest dialog/panel
                let root = heading;
                for (let i = 0; i < 8 && root.parentElement; i++) {
                    root = root.parentElement;
                    if (root.getAttribute('role') === 'dialog') break;
                }

                const candidates = [root, ...root.querySelectorAll('*')];
                let best = null;
                let bestArea = -1;

                for (const candidate of candidates) {
                    const overflowY = window.getComputedStyle(candidate).overflowY;
                    if (overflowY !== 'auto' && overflowY !== 'scroll') continue;

                    const extraHeight = candidate.scrollHeight - candidate.clientHeight;
                    if (extraHeight <= 5) continue;

                    if (extraHeight > bestArea) {
                        bestArea = extraHeight;
                        best = candidate;
                    }
                }
                return best;
                """,
                popup_heading,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Field extractors
    # ------------------------------------------------------------------

    def extract_restaurant(self, article) -> Optional[str]:
        # The profile image alt text is the business name.
        try:
            for img in article.find_elements(By.XPATH, ".//img[@alt]"):
                alt = self.safe_attribute(img, "alt")
                if alt and alt.strip() and alt.lower() not in {"image", "photo", "logo"}:
                    return alt.strip()
        except Exception:
            pass

        # Header fallback
        try:
            for heading in article.find_elements(By.XPATH, ".//*[@role='heading']"):
                text = self.safe_text(heading)
                if text and "Updates" not in text:
                    return text
        except Exception:
            pass

        return None

    def extract_date(self, article) -> Optional[str]:
        """Dates look like 'Jul 28, 2025' / 'Dec 5, 2025'."""
        date_patterns = [
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        ]

        try:
            for element in article.find_elements(By.XPATH, ".//*"):
                text = self.safe_text(element)
                if not text or len(text) > 100:   # prefer small elements
                    continue
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(0)
        except Exception:
            pass

        return None

    def extract_title(self, article) -> Optional[str]:
        """span[role='heading'][aria-level='3'] inside the post."""
        try:
            for heading in article.find_elements(
                By.XPATH, ".//*[@role='heading' and @aria-level='3']"
            ):
                text = self.safe_text(heading)
                if text:
                    return text
        except Exception:
            pass
        return None

    def extract_validity(self, article) -> Optional[str]:
        """'Valid 5 Dec - 21 Dec' style line."""
        try:
            for element in article.find_elements(By.XPATH, ".//*"):
                text = self.safe_text(element)
                if text and re.match(r"^Valid\b", text, re.IGNORECASE):
                    return text
        except Exception:
            pass
        return None

    def click_more_if_present(self, article):
        """Expand the post body when Google has collapsed it behind More.

        Google has used several DOM shapes for this control. We deliberately
        search by visible text/ARIA instead of relying on one generated
        jsname, then re-find the article after the click because Google often
        replaces the node.
        """
        selectors = [
            (By.XPATH, ".//*[@jsname='WUPT1e' and @role='button']"),
            (By.XPATH, ".//*[@role='button' and normalize-space()='More']"),
            (By.XPATH, ".//button[normalize-space()='More']"),
            (By.XPATH, ".//a[normalize-space()='More']"),
            (By.XPATH, ".//*[@aria-label='More']"),
            (By.XPATH, ".//*[@title='More']"),
            (By.XPATH, ".//*[normalize-space()='More']"),
        ]

        seen = set()
        for by, selector in selectors:
            try:
                candidates = article.find_elements(by, selector)
            except Exception:
                continue

            for candidate in candidates:
                try:
                    if not self.visible(candidate):
                        continue

                    # If the text is inside a span/div, click its actual
                    # button/link parent when one exists.
                    clickable = self.driver.execute_script(
                        """
                        const e = arguments[0];
                        return e.closest('button,[role=button],a') || e;
                        """,
                        candidate,
                    )
                    key = id(clickable)
                    if key in seen:
                        continue
                    seen.add(key)

                    before = self.safe_text(article)
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center',inline:'nearest'});",
                        clickable,
                    )
                    self.sleep(0.2)

                    try:
                        clickable.click()
                    except Exception:
                        self.driver.execute_script("arguments[0].click();", clickable)

                    # Wait briefly for Google's re-render/expanded text.
                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        try:
                            post_id = self.safe_attribute(article, "data-post-id")
                            fresh = self.refind_post(post_id, attempts=1) if post_id else None
                            if fresh is not None:
                                article = fresh
                            after = self.safe_text(article)
                            if len(after) > len(before) or "More" not in after[-80:]:
                                break
                        except Exception:
                            pass
                        self.sleep(0.15)
                    return True
                except (StaleElementReferenceException, WebDriverException):
                    continue
                except Exception:
                    continue

        return False

    def extract_content(self, article) -> Optional[str]:
        """Return the fullest available post copy after any More expansion."""
        candidates = []

        # Google has used both jsname and data-content-id for the body.
        for selector in [
            ".//*[@jsname='g3HNze']",
            ".//*[@data-content-id]",
        ]:
            try:
                for element in article.find_elements(By.XPATH, selector):
                    text = self.safe_text(element)
                    if text and len(text) >= 10:
                        candidates.append(text)
            except Exception:
                continue

        if candidates:
            # The body node is normally the longest of these candidates.
            return max(candidates, key=len)

        # Fallback: collect visible text blocks, preferring the longest block
        # that is not just a UI label. This catches cards whose body markup
        # differs from the common Google structure.
        excluded = {"More", "Share", "Like", "Comment", "Copy link"}
        try:
            blocks = []
            for element in article.find_elements(By.XPATH, ".//*[self::span or self::div or self::p]"):
                text = self.safe_text(element)
                if not text or text in excluded or len(text) < 20:
                    continue
                blocks.append(text)
            if blocks:
                return max(blocks, key=len)
        except Exception:
            pass

        return None

    def extract_image(self, article, restaurant: Optional[str] = None) -> Optional[str]:
        """
        Real post media lives inside an element carrying data-media-index.
        The profile logo is never inside one, so it is never picked up.
        """
        try:
            if restaurant is None:
                restaurant = self.extract_restaurant(article) or ""

            for img in article.find_elements(By.XPATH, ".//*[@data-media-index]//img"):
                src = self.safe_attribute(img, "src")
                if not src or src.startswith("data:"):
                    continue

                alt = self.safe_attribute(img, "alt") or ""
                if restaurant and alt.strip().lower() == restaurant.strip().lower():
                    continue

                return src
        except Exception:
            pass

        return None

    def extract_video(self, article) -> Optional[str]:
        """
        Extract the actual video URL used by Google.

        Google often exposes a media resolver URL such as:

            https://lh3.googleusercontent.com/geougc/...

        Opening that URL in Chrome causes Google to redirect it to a
        temporary signed ``googlevideo.com/videoplayback`` URL.  We prefer
        that resolved URL because it is directly playable by an HTML video
        element.  If Google does not expose the redirect, the original
        resolver URL is retained as a fallback.
        """

        fallback_urls = []

        # ------------------------------------------------------------
        # 1. Find the media URL exposed by the current post.
        # ------------------------------------------------------------
        selectors = [
            ".//*[@data-media-index]//video",
            ".//*[@data-media-index]//source",
            ".//*[@data-media-index]//a[@href]",
        ]

        for selector in selectors:
            try:
                elements = article.find_elements(By.XPATH, selector)
            except Exception:
                continue

            for element in elements:
                for attribute in [
                    "currentSrc",
                    "src",
                    "href",
                    "data-src",
                    "data-video-url",
                ]:
                    try:
                        if attribute == "currentSrc":
                            value = self.driver.execute_script(
                                "return arguments[0].currentSrc || '';",
                                element,
                            )
                        else:
                            value = self.safe_attribute(element, attribute)
                    except Exception:
                        value = None

                    if not value:
                        continue

                    value = str(value).strip()
                    if not value or value.startswith("blob:"):
                        continue

                    # Already resolved by Chrome.
                    if "googlevideo.com/videoplayback" in value:
                        return value

                    if value not in fallback_urls:
                        fallback_urls.append(value)

        if not fallback_urls:
            return None

        # ------------------------------------------------------------
        # 2. Open the resolver URL in a temporary Chrome tab.
        #    This reproduces what happens when you manually click the
        #    lh3.googleusercontent.com video URL.
        # ------------------------------------------------------------
        original_handle = None
        temp_handle = None

        try:
            original_handle = self.driver.current_window_handle
            original_handles = set(self.driver.window_handles)

            resolver_url = fallback_urls[0]

            self.driver.execute_script(
                "window.open(arguments[0], '_blank');",
                resolver_url,
            )

            # Wait for Chrome to create the temporary tab.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                handles = set(self.driver.window_handles)
                new_handles = handles - original_handles
                if new_handles:
                    temp_handle = next(iter(new_handles))
                    break
                time.sleep(0.1)

            if temp_handle is not None:
                self.driver.switch_to.window(temp_handle)

                # The resolver normally redirects very quickly.  Give it a
                # few seconds because large Google media requests can be slow.
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    try:
                        current_url = self.driver.current_url or ""
                    except Exception:
                        current_url = ""

                    if "googlevideo.com/videoplayback" in current_url:
                        return current_url

                    time.sleep(0.2)

        except Exception as exc:
            # Resolver failures should never make the whole post fail.
            print(
                f"    Video resolver fallback failed: "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            # Always close the temporary media tab and return to the Updates
            # popup.  This is important because the rest of the scraper must
            # continue operating on the original Google page.
            try:
                if temp_handle is not None:
                    self.driver.close()
            except Exception:
                pass

            try:
                if original_handle is not None and original_handle in self.driver.window_handles:
                    self.driver.switch_to.window(original_handle)
            except Exception:
                pass

        # ------------------------------------------------------------
        # 3. Last chance: Chrome may already have exposed the final URL
        #    through the video element's currentSrc.
        # ------------------------------------------------------------
        for selector in [
            ".//*[@data-media-index]//video",
            ".//*[@data-media-index]//source",
        ]:
            try:
                for element in article.find_elements(By.XPATH, selector):
                    try:
                        current_src = self.driver.execute_script(
                            "return arguments[0].currentSrc || arguments[0].src || '';",
                            element,
                        )
                    except Exception:
                        current_src = ""

                    if current_src and "googlevideo.com/videoplayback" in current_src:
                        return current_src
            except Exception:
                continue

        # Google did not expose the final signed URL.  Keep the resolver URL
        # rather than losing the video completely.
        return fallback_urls[0]

    def extract_cta(self, article) -> Optional[str]:
        """
        Best-effort call-to-action label ("Book", "Learn more", ...).
        Not part of test7.py - text-matches the button against CTA_LABELS so
        ScrapedPost.cta_text keeps getting filled. Returns None if absent.
        """
        wanted = {label.lower() for label in CTA_LABELS}
        try:
            for xpath in [".//a[@href]", ".//*[@role='button']"]:
                for element in article.find_elements(By.XPATH, xpath):
                    text = self.safe_text(element)
                    if text and text.lower() in wanted:
                        return text
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # PHASE 1 - capture a card's data
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_post_text(value: Optional[str]) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).casefold()

    def _post_fingerprint(self, post: GooglePost) -> str:
        """Stable fingerprint of the visible post content, not Google's ID."""
        parts = [
            self._normalise_post_text(post.title),
            self._normalise_post_text(post.validity),
            self._normalise_post_text(post.content),
        ]
        text_key = "|".join(parts)
        key = text_key if text_key.strip("|") else (post.image_url or post.video_url or "")
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def capture_post_data(self, article) -> Optional[GooglePost]:
        """Reads everything present in the card. Never opens Share."""
        post_id = None

        try:
            post_id = self.safe_attribute(article, "data-post-id")
            if not post_id:
                return None

            # Bring this exact card into view so Google finishes rendering it.
            try:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center', inline:'nearest'});",
                    article,
                )
            except Exception:
                pass
            self.sleep(0.35)

            # "More" can expand the text and re-render the article node.
            self.click_more_if_present(article)
            self.sleep(0.25)

            fresh = self.refind_post(post_id)
            if fresh is not None:
                article = fresh

            feature_id = self.safe_attribute(article, "data-feature-id")
            content_id = self.safe_attribute(article, "data-content-id")

            if not content_id:
                try:
                    nested = article.find_elements(By.XPATH, ".//*[@data-content-id]")
                    if nested:
                        content_id = self.safe_attribute(nested[0], "data-content-id")
                except Exception:
                    pass

            restaurant = self.extract_restaurant(article)
            image_url = self.extract_image(article, restaurant)
            video_url = None if image_url else self.extract_video(article)

            return GooglePost(
                restaurant=restaurant,
                date=self.extract_date(article),
                title=self.extract_title(article),
                validity=self.extract_validity(article),
                content=self.extract_content(article),
                image_url=image_url,
                video_url=video_url,
                post_id=post_id,
                feature_id=feature_id,
                content_id=content_id,
                post_url=None,
                cta_text=self.extract_cta(article),
            )

        except Exception as exc:
            print(f"    Capture failed for post {post_id or 'unknown'}: {type(exc).__name__}: {exc}")
            return None

    def collect_current_posts(self, supplied_posts=None):
        """Capture only genuinely new IDs/content from the currently rendered cards."""
        posts = supplied_posts if supplied_posts is not None else self.get_posts()
        if not posts:
            return 0

        post_ids = []
        for article in posts:
            post_id = self.safe_attribute(article, "data-post-id")
            if post_id and post_id not in post_ids:
                post_ids.append(post_id)

        added = 0
        for post_id in post_ids:
            # Google keeps the same DOM cards around for many scroll passes.
            # Do not re-open/re-extract an already seen ID. This is what was
            # causing the repeated "Skipping repeated post" loop.
            if post_id in self.seen_post_ids:
                continue
            self.seen_post_ids.add(post_id)

            article = next(
                (candidate for candidate in posts
                 if self.safe_attribute(candidate, "data-post-id") == post_id),
                None,
            )
            if article is None:
                article = self.refind_post(post_id)
            if article is None:
                continue

            captured = None
            for attempt in range(2):
                try:
                    captured = self.capture_post_data(article)
                    if captured is not None:
                        break
                except Exception:
                    pass
                if attempt == 0:
                    article = self.refind_post(post_id)

            if captured is None:
                continue

            fingerprint = self._post_fingerprint(captured)
            if fingerprint in self.collected_fingerprints:
                self.duplicate_hits += 1
                print(
                    f"    Skipping duplicate content: {post_id} "
                    f"(duplicate {self.duplicate_hits}/{MAX_DUPLICATE_HITS})"
                )
                if self.duplicate_hits >= MAX_DUPLICATE_HITS:
                    self.stop_requested = True
                    print("[*] First 3 duplicate post contents found. Stopping scrape early to save time.")
                    break
                continue

            self.posts_checked += 1

            # Already saved from a previous scrape? Posts come newest-first, so
            # MAX_KNOWN_STREAK saved posts in a row means nothing older is new.
            if self.is_known is not None:
                item = _post_to_item(self.competitor_name, captured)
                if item is not None and self.is_known(item):
                    self.known_streak += 1
                    self.known_total += 1
                    # Remember the fingerprint so scroll-progress logic still
                    # sees movement, but don't keep the post (no URL work for it).
                    self.collected_fingerprints.add(fingerprint)
                    print(
                        f"    Already saved: {post_id} "
                        f"(in a row {self.known_streak}/{MAX_KNOWN_STREAK})"
                    )
                    if self.known_streak >= MAX_KNOWN_STREAK:
                        self.up_to_date = True
                        self.stop_requested = True
                        print(
                            f"[+] {MAX_KNOWN_STREAK} already-saved posts in a row. "
                            "Posts are up to date - stopping the scrape early."
                        )
                        break
                    continue
                self.known_streak = 0

            self.collected[post_id] = captured
            self.collected_fingerprints.add(fingerprint)
            added += 1
            print(f"    Captured NEW post {len(self.collected):02d}: {post_id}")

        return added

    def _scroll_metrics(self, container):
        return self.driver.execute_script(
            "return [arguments[0].scrollTop, arguments[0].scrollHeight, arguments[0].clientHeight];",
            container,
        )

    def scroll_all_posts(self) -> Dict[str, GooglePost]:
        """Scroll until Google stops revealing new cards, with hard progress guards."""
        print("[*] Loading all available posts...")

        initial_posts = self.wait_for_post_cards(timeout=12)
        if initial_posts:
            print(f"[*] Initial post cards detected: {len(initial_posts)}")
            self.collect_current_posts(initial_posts)
            if self.stop_requested:
                print(f"[+] Finished loading posts. New posts: {len(self.collected)}")
                return self.collected
        else:
            print("[!] No article[data-post-id] cards yet. Waiting once more...")
            self.sleep(2)

        container = None
        for _ in range(10):
            container = self.find_scroll_container()
            if container is not None:
                break
            self.sleep(0.5)

        if container is None:
            print("[!] Could not identify the popup's scroll container.")
            self.scroll_using_javascript()
            return self.collected

        deadline = time.monotonic() + SCROLL_WALL_CLOCK_BUDGET
        stalled_rounds = 0
        last_top = -1
        last_height = -1

        for round_number in range(1, MAX_SCROLL_ROUNDS + 1):
            if time.monotonic() > deadline:
                print(f"[!] Scroll budget of {SCROLL_WALL_CLOCK_BUDGET}s reached - stopping.")
                break

            fresh_container = self.find_scroll_container()
            if fresh_container is not None:
                container = fresh_container

            before_count = len(self.collected_fingerprints)
            self.collect_current_posts(self.get_posts())
            if self.stop_requested:
                break

            try:
                top, height, client = self._scroll_metrics(container)
                if height <= client:
                    print("[*] Updates list is not scrollable. Finished.")
                    break

                self.driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight;", container
                )
                self.sleep(SCROLL_PAUSE)

                # Google may replace the scroll container after lazy rendering.
                fresh_container = self.find_scroll_container()
                if fresh_container is not None:
                    container = fresh_container
                after_top, after_height, after_client = self._scroll_metrics(container)
                self.collect_current_posts(self.get_posts())
                after_count = len(self.collected_fingerprints)
                if self.stop_requested:
                    break

                print(
                    f"    Scroll {round_number:02d} | posts={after_count} | "
                    f"top={int(after_top)} | height={int(after_height)}"
                )

                made_progress = (after_count > before_count or
                                 after_top > last_top + 2 or
                                 after_height > last_height + 2)
                at_bottom = after_top + after_client >= after_height - 10

                if made_progress:
                    stalled_rounds = 0
                else:
                    stalled_rounds += 1

                last_top, last_height = after_top, after_height

                # Once we're at the bottom and two consecutive passes reveal
                # neither new content nor a larger list, there is nothing more
                # to lazy-load. Never keep looping over the same cards.
                if at_bottom and stalled_rounds >= 2:
                    print("[*] Reached bottom with no new posts. Finished.")
                    break

                if stalled_rounds >= 3:
                    print("[*] Scroll position/content stopped changing. Finished.")
                    break

            except Exception as exc:
                print(f"    Scroll error: {exc}")
                break

        print(f"[+] Finished loading posts. Total unique posts: {len(self.collected_fingerprints)}")
        return self.collected

    def scroll_using_javascript(self):
        """Fallback if the exact scroll container cannot be found."""
        print("[*] Using fallback popup scrolling...")

        stable_rounds = 0
        last_count = 0

        for round_number in range(1, MAX_SCROLL_ROUNDS + 1):
            self.collect_current_posts()
            if self.stop_requested:
                break
            count_before = len(self.collected)

            self.driver.execute_script(
                """
                const root = document.querySelector('[role="dialog"]') || document.body;
                let target = root;

                for (const el of [root, ...root.querySelectorAll('*')]) {
                    const style = window.getComputedStyle(el);
                    if ((style.overflowY === 'auto' || style.overflowY === 'scroll') &&
                        el.scrollHeight > el.clientHeight) {
                        target = el;
                    }
                }
                target.scrollTop = target.scrollHeight;
                """
            )

            self.sleep(SCROLL_PAUSE)
            self.collect_current_posts()
            if self.stop_requested:
                break

            count_after = len(self.collected)
            print(f"    Fallback scroll {round_number:02d} | posts={count_after}")

            if count_after <= count_before and count_after <= last_count:
                stable_rounds += 1
            else:
                stable_rounds = 0

            last_count = count_after

            if stable_rounds >= NO_NEW_POST_ROUNDS:
                break

        print(f"[+] Finished loading posts. Total unique posts: {len(self.collected)}")

    # ------------------------------------------------------------------
    # PHASE 2 - share.google URLs
    # ------------------------------------------------------------------

    def find_share_popup_url(self) -> Optional[str]:
        """Share popup: <a jsname="RYUcpc" role="button" href="https://share.google/...">"""
        selectors = [
            (By.XPATH, "//a[@jsname='RYUcpc' and @href]"),
            (
                By.XPATH,
                "//div[contains(@class,'NTNSUb')]//a[@href and contains(@href,'share.google')]",
            ),
            (By.XPATH, "//a[contains(@href,'share.google')]"),
        ]

        for by, selector in selectors:
            try:
                for element in self.driver.find_elements(by, selector):
                    if not self.visible(element):
                        continue
                    href = self.safe_attribute(element, "href")
                    if href:
                        return href
            except Exception:
                continue

        return None

    def close_share_popup(self):
        selectors = [
            (By.XPATH, "//button[@aria-label='Close']"),
            (By.XPATH, "//*[@role='button' and @aria-label='Close']"),
        ]

        for by, selector in selectors:
            try:
                for button in self.driver.find_elements(by, selector):
                    if self.visible(button):
                        self.click_element(button)
                        self.sleep(0.4)
                        return True
            except Exception:
                continue

        try:
            self.driver.switch_to.active_element.send_keys(Keys.ESCAPE)
            self.sleep(0.4)
            return True
        except Exception:
            return False

    def extract_post_url_from_share(self, article) -> Optional[str]:
        """Opens Share for THIS article and reads https://share.google/..."""
        share_button = None

        selectors = [
            (By.XPATH, ".//*[@role='button' and @aria-label='Share']"),
            (By.XPATH, ".//*[@aria-label='Share']"),
            (By.XPATH, ".//*[@jsname='YOuPgf']"),
        ]

        for by, selector in selectors:
            try:
                for button in article.find_elements(by, selector):
                    if self.visible(button):
                        share_button = button
                        break
                if share_button is not None:
                    break
            except Exception:
                continue

        if share_button is None:
            print("    Share button not found.")
            return None

        print("    Opening Share...")

        if not self.click_element(share_button):
            print("    Could not click Share.")
            return None

        for _ in range(20):
            url = self.find_share_popup_url()
            if url:
                print(f"    Share URL: {url}")
                self.close_share_popup()
                return url
            self.sleep(0.25)

        print("    Share URL not found.")
        self.close_share_popup()
        return None

    def locate_post(self, post_id: str):
        """
        Find a card by id, re-opening the popup if Share closed it.

        After a popup reopen the list starts at the top. Cards further down
        may not be rendered yet, so if the direct lookup misses we scroll the
        popup down in steps until the card appears (small addition to test7
        so long feeds don't lose their later URLs).
        """
        article = self.refind_post(post_id, attempts=4)
        if article is not None:
            return article

        container = self.find_scroll_container()
        if container is None:
            return None

        try:
            self.driver.execute_script("arguments[0].scrollTop = 0;", container)
            self.sleep(0.5)

            for _ in range(MAX_SCROLL_ROUNDS):
                self.driver.execute_script(
                    "arguments[0].scrollTop += Math.max(arguments[0].clientHeight * 0.8, 200);",
                    container,
                )
                self.sleep(0.8)

                article = self.refind_post(post_id, attempts=2)
                if article is not None:
                    return article

                top, height, client = self._scroll_metrics(container)
                if top + client >= height - 10:
                    break
        except Exception:
            pass

        return None

    def prepare_post(self, post_id: str):
        """Bring a captured post into view so its Share button is usable."""
        if not self.ensure_updates_popup_open():
            return None

        article = self.locate_post(post_id)
        if article is None:
            return None

        try:
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block:'center', inline:'nearest'});",
                article,
            )
        except Exception:
            pass

        self.sleep(0.7)
        return self.refind_post(post_id)

    def enrich_post_urls(self, posts: List[GooglePost]) -> List[GooglePost]:
        """PHASE 2: one Share URL per post, one post at a time."""
        section("PHASE 2: COLLECTING SHARE URLS")

        total = len(posts)

        for index, post in enumerate(posts, start=1):
            print(f"\n[{index:02d}/{total:02d}] post {post.post_id}")

            if not post.post_id:
                print("    Missing post ID. Skipping URL.")
                continue

            article = None

            for attempt in range(3):
                try:
                    article = self.prepare_post(post.post_id)
                    if article is not None:
                        break
                except (StaleElementReferenceException, NoSuchElementException):
                    pass
                except CaptchaBlocked:
                    raise
                except Exception as exc:
                    print(f"    Attempt {attempt + 1} error: {exc}")

                if attempt < 2:
                    self.sleep(0.5)

            if article is None:
                print("    Could not find post for Share URL.")
                continue

            try:
                url = self.extract_post_url_from_share(article)

                if url and url.startswith("https://share.google/"):
                    post.post_url = url
                else:
                    post.post_url = None
                    if url:
                        print(f"    Ignoring non-direct post URL: {url}")
            except Exception as exc:
                print(f"    URL extraction failed: {exc}")
                post.post_url = None

        got = sum(1 for p in posts if p.post_url)
        print(f"\n[+] Finished Share URL collection: {got}/{total} posts have a URL.")
        return posts

    # ------------------------------------------------------------------
    # Full flow
    # ------------------------------------------------------------------

    def scrape_posts(self) -> List[GooglePost]:
        """Assumes the Updates popup is open (see open_updates())."""
        if self.find_updates_popup() is None:
            if not self.ensure_updates_popup_open():
                raise RuntimeError("Updates popup is not open.")

        section("PHASE 1: READING POST CARDS")
        self.scroll_all_posts()

        if not self.collected:
            if self.up_to_date:
                print("[+] Nothing new - posts are up to date.")
            else:
                print("[-] No posts were collected during scrolling.")
            return []

        posts = list(self.collected.values())
        print(f"[+] Collected card data for {len(posts)} posts.")

        self.enrich_post_urls(posts)
        return posts


# =====================================================================
# Converting scraped posts into the dict shape main.py stores
# =====================================================================

def _normalise_post_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _post_content_fingerprint(post: GooglePost) -> str:
    """Hash the actual post copy so a new Google ID cannot create a duplicate row."""
    text_key = "|".join([
        _normalise_post_text(post.title),
        _normalise_post_text(post.validity),
        _normalise_post_text(post.content),
    ])
    key = text_key if text_key.strip("|") else (post.image_url or post.video_url or "")
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def _make_content_hash(competitor_name: str, post: GooglePost) -> str:
    # Include the competitor, but deliberately do not include Google's post ID.
    # The same visible post can appear with a different transient ID.
    key = f"{_normalise_post_text(competitor_name)}|{_post_content_fingerprint(post)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def _post_to_item(competitor_name: str, p: GooglePost) -> Optional[Dict[str, Any]]:
    """GooglePost -> the dict shape main.py stores. None if the post is empty."""
    content = (p.content or "").strip()
    title = (p.title or "").strip() or None

    if not content and not title and not p.image_url and not p.video_url:
        return None

    return {
        "date": p.date or "",
        "title": title,
        "validity": (p.validity or "").strip() or None,
        "content": content or (title or ""),
        "cta_text": p.cta_text,
        "image_url": p.image_url,
        "video_url": p.video_url,
        "post_url": p.post_url,
        "post_id": p.post_id,
        "feature_id": p.feature_id,
        "content_id": p.content_id,
        "content_hash": _make_content_hash(competitor_name, p),
    }


def _to_db_posts(competitor_name: str, gposts: List[GooglePost]) -> List[Dict[str, Any]]:
    posts = []
    seen_hashes = set()

    for p in gposts:
        item = _post_to_item(competitor_name, p)
        if item is None:
            continue
        if item["content_hash"] in seen_hashes:
            continue
        seen_hashes.add(item["content_hash"])
        posts.append(item)

    return posts


def run_competitor_scrape(
    competitor_name: str,
    maps_url: Optional[str] = None,
    is_known: Optional[Callable[[Dict[str, Any]], bool]] = None,
    stats: Optional[Dict[str, Any]] = None,
    address: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Public entry point main.py imports.

    address:  the branch's street address. The Google search uses "name + address"
              so chains with many branches open the RIGHT one. If it is missing it
              is read from maps_url first (and returned in stats["address"]).

    is_known: optional callback(post_dict) -> bool telling us whether a post is
              already saved. When MAX_KNOWN_STREAK saved posts show up in a row
              the scrape stops early ("posts are up to date") and only the NEW
              posts found before that point are returned.
    stats:    optional dict that gets filled with
              {"up_to_date": bool, "posts_checked": int, "known_posts": int}

    Searches Google for the competitor, opens "View previous updates on
    Google", reads every post card, then collects each post's share.google
    URL. Returns a list of dicts:

        date, title, validity, content, cta_text, image_url, video_url,
        post_url, post_id, feature_id, content_id, content_hash
    """
    driver = init_driver()

    # Watchdog: force-closes the browser if this competitor blows through its
    # wall-clock budget, so one bad page can never hang the whole batch.
    watchdog_fired = threading.Event()

    def _watchdog():
        watchdog_fired.set()
        print(f"[!] {competitor_name}: exceeded {COMPETITOR_WALL_CLOCK_BUDGET}s budget - force-closing browser.")
        try:
            driver.quit()
        except Exception:
            pass

    timer = threading.Timer(COMPETITOR_WALL_CLOCK_BUDGET, _watchdog)
    timer.daemon = True
    timer.start()

    scraper = GoogleUpdatesScraper(driver, competitor_name=competitor_name, is_known=is_known)
    gposts: List[GooglePost] = []

    resolved_address = (address or "").strip() or None

    def _fill_stats():
        if stats is not None:
            stats["address"] = resolved_address
            stats["up_to_date"] = scraper.up_to_date
            stats["posts_checked"] = scraper.posts_checked
            stats["known_posts"] = scraper.known_total

    try:
        print("\n" + "#" * 80)
        print(f"SCRAPING: {competitor_name}")
        print("#" * 80)

        # Chains (e.g. "Jimmy's Burger") have many branches. Make sure we know THIS
        # branch's address so Google opens the right panel.
        if not resolved_address and maps_url:
            section("LOOKING UP BRANCH ADDRESS")
            try:
                driver.get(maps_url)
                scraper.sleep(4)
                dismiss_consent_dialog(driver)
                resolved_address = _read_place_address(driver)
                print(f"[+] Branch address: {resolved_address or 'not found'}")
            except Exception as exc:
                print(f"[!] Could not read the branch address from the Maps link: {exc}")

        search_query = build_search_query(competitor_name, resolved_address)

        section("GOOGLE SEARCH")
        try:
            scraper.search(search_query)
        except TimeoutException:
            print(f"[-] {competitor_name}: Google search timed out. Skipping.")
            if stats is not None:
                stats["error"] = "Google search timed out. Try again."
            return []

        section("OPENING UPDATES")
        try:
            scraper.open_updates()
        except NoUpdatesFound as e:
            print(f"[-] {e}")
            if stats is not None:
                stats["no_updates"] = True
            return []
        except RuntimeError as e:
            print(f"[-] {competitor_name}: {e} Skipping.")
            if stats is not None:
                stats["error"] = str(e)
            return []

        gposts = scraper.scrape_posts()
        if not gposts:
            if scraper.up_to_date:
                print(f"[+] {competitor_name}: posts are up to date, nothing new to add.")
            else:
                print(f"[-] {competitor_name}: no update cards found.")
            return []

    except CaptchaBlocked:
        gposts = list(scraper.collected.values())
        print(f"[-] {competitor_name}: CAPTCHA not solved in time. "
              f"Keeping {len(gposts)} post(s) collected so far.")
    except Exception as e:
        # Includes the watchdog killing the driver mid-call. Keep whatever
        # Phase 1 already collected instead of throwing the whole run away.
        gposts = list(scraper.collected.values())
        reason = "watchdog timeout" if watchdog_fired.is_set() else f"{e.__class__.__name__}: {e}"
        print(f"[-] {competitor_name}: stopped early ({reason}). "
              f"Keeping {len(gposts)} post(s) collected so far.")
    finally:
        timer.cancel()
        _fill_stats()
        try:
            driver.quit()
        except Exception:
            pass

    # Sanity check: the panel we scraped should belong to the competitor we
    # picked (feature_id == the place id inside the Maps URL).
    expected_id = extract_place_id(maps_url) if maps_url else None
    if expected_id:
        wrong = [p for p in gposts if p.feature_id and p.feature_id.lower() != expected_id.lower()]
        if wrong:
            print(f"[!] {competitor_name}: {len(wrong)} post(s) have a feature_id that does not match "
                  f"the competitor's place id ({expected_id}). Google may have shown a different business.")

    posts = _to_db_posts(competitor_name, gposts)
    print(f"[+] Successfully collected {len(posts)} posts for {competitor_name}.")
    return posts