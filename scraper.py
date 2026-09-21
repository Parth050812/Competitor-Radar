import os
import time
import re
import hashlib
import threading
import queue
from typing import List, Dict, Any, Optional
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
SCROLL_WALL_CLOCK_BUDGET = 90   # seconds - hard cap on total time spent scrolling one dialog
COMPETITOR_WALL_CLOCK_BUDGET = 240  # seconds - hard cap per competitor before we give up and move on

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

            candidates.append({
                "name": name,
                "maps_url": href,
                "place_id": extract_place_id(href),
            })

        return candidates

    finally:
        driver.quit()


def _get_updates_dialog(driver):
    date_re = re.compile(r"[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}")

    headings = driver.find_elements(
        By.XPATH,
        '//*[contains(text(), "Updates from") or contains(@aria-label, "Updates from")]'
    )
    if not headings:
        return None

    heading = headings[0]
    ancestors = heading.find_elements(By.XPATH, "ancestor::div")
    ancestors.reverse()

    fallback = None
    for anc in ancestors:
        try:
            text = anc.text.strip()
        except Exception:
            continue
        if not text:
            continue
        if fallback is None:
            fallback = anc

        date_hits = len(date_re.findall(text))
        cta_hits = sum(text.count(cta) for cta in CTA_LABELS)
        if date_hits >= 2 or cta_hits >= 2 or (date_hits >= 1 and cta_hits >= 1):
            return anc

    return fallback


IMG_SIZE_PATTERN = re.compile(r"=w(\d+)-h(\d+)")


def _image_size(src: str):
    """Pull the =w###-h### hint Google appends to these URLs, if present."""
    m = IMG_SIZE_PATTERN.search(src)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def pick_post_image(valid_imgs: List[str]) -> Optional[str]:
    """
    Pick the actual post/media image out of a card's <img> srcs, skipping the
    business logo/avatar. The logo is near-square and small (Google renders it
    around 40-96px, e.g. '=w48-h48-p-k-no-il'); the real post image is usually
    much wider/rectangular (e.g. '=w426-h240-...').

    Falls back to "first image after the first one" (old index[1] behaviour)
    if none of the URLs carry a recognizable size hint.
    """
    if not valid_imgs:
        return None

    scored = []
    for src in valid_imgs:
        size = _image_size(src)
        scored.append((src, size))

    def is_logo_like(size):
        if size is None:
            return False
        w, h = size
        # small and roughly square -> avatar/logo icon
        return w <= 96 and h <= 96 and abs(w - h) <= 4

    non_logo = [src for src, size in scored if not is_logo_like(size)]
    if non_logo:
        # Prefer the largest candidate (post media is usually the biggest image in the card)
        def area(src):
            size = _image_size(src)
            return (size[0] * size[1]) if size else 0
        return max(non_logo, key=area)

    # No size hints at all / everything looked logo-sized: fall back to
    # "the image right after the first" (the old heuristic).
    if len(valid_imgs) > 1:
        return valid_imgs[1]
    return None


def scroll_modal_container(driver, on_scroll=None, max_stall_rounds: int = 3):
    """
    Dynamically scrolls the updates panel container until no new content loads
    (Infinite Scroll), ensuring all lazy-loaded posts are fetched.

    IMPORTANT: Google's updates feed is virtualized - once you've scrolled past
    them, earlier cards can be *removed* from the DOM to make room for newly
    loaded ones, so scrollHeight can plateau/repeat even while new posts are
    still arriving beneath the fold, and cards seen early are gone by the time
    you read the DOM at the end. That's why only ~7 were being captured.

    To fix this, pass `on_scroll(driver, dialog)` - it's called after every
    scroll step so posts can be extracted and accumulated incrementally,
    before they get evicted. Height is used only as a stopping condition,
    not as the sole thing tracked, and we require several stalled rounds in a
    row (not just one) before concluding there's nothing left.
    """
    def _safe_script(script, *args, default=None):
        """execute_script has a hard SCRIPT_TIMEOUT set on the driver, so a
        wedged page can't hang this forever - but we still guard the call so
        one failed script doesn't kill the whole scroll loop."""
        try:
            return driver.execute_script(script, *args)
        except (TimeoutException, WebDriverException):
            return default

    try:
        dialog = _get_updates_dialog(driver)
        if dialog is None:
            return

        last_height = -1
        stall_rounds = 0
        max_iterations = 200  # secondary safety cap, but the wall-clock budget below binds first
        iterations = 0
        deadline = time.monotonic() + SCROLL_WALL_CLOCK_BUDGET

        while iterations < max_iterations and time.monotonic() < deadline:
            iterations += 1

            # Step scroll in smaller increments rather than jumping straight to the
            # bottom - Google's lazy loader can miss the "near bottom" trigger on a
            # single huge jump, especially right after the dialog opens.
            _safe_script(
                "arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].clientHeight * 0.8;",
                dialog,
            )
            time.sleep(1.2)

            if on_scroll is not None:
                try:
                    on_scroll(driver, dialog)
                except Exception:
                    pass

            new_height = _safe_script("return arguments[0].scrollHeight", dialog, default=last_height)
            at_bottom = _safe_script(
                "return arguments[0].scrollTop + arguments[0].clientHeight >= arguments[0].scrollHeight - 5;",
                dialog,
                default=True,
            )

            if new_height == last_height and at_bottom:
                stall_rounds += 1
                time.sleep(1.5)  # give a slow-rendering feed a chance to catch up
                if on_scroll is not None:
                    try:
                        on_scroll(driver, dialog)
                    except Exception:
                        pass
                if stall_rounds >= max_stall_rounds:
                    break
            else:
                stall_rounds = 0
            last_height = new_height

        if time.monotonic() >= deadline:
            print(f"[!] Scroll budget ({SCROLL_WALL_CLOCK_BUDGET}s) hit - moving on with what was collected so far.")
    except Exception:
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
        except Exception:
            pass



def _find_post_container(panel, min_child_text_len: int = 15):
    all_elements = panel.find_elements(By.XPATH, ".//*")
    best_children = []

    for el in all_elements:
        try:
            children = el.find_elements(By.XPATH, "./*")
        except Exception:
            continue
        if len(children) < 2:
            continue

        substantial = []
        for c in children:
            try:
                t = c.text.strip()
            except Exception:
                t = ""
            if len(t) >= min_child_text_len:
                substantial.append(c)

        if len(substantial) >= 2 and len(substantial) > len(best_children):
            best_children = substantial

    return best_children


def _extract_card(card, competitor_name: str) -> Optional[Dict[str, Any]]:
    try:
        text = card.text.strip()
        if not text or len(text) < 20:
            return None

        if text.count("Recent updates") > 1:
            return None
        if text.strip() == competitor_name.strip():
            return None
        if "Updates from" in text and len(text) < 60:
            return None
        if len(text) > 1500:
            return None

        # 1. Extract video links/embeds and append them into the content text
        video_links = []
        try:
            links = card.find_elements(By.XPATH, './/a[@href]')
            for link in links:
                href = link.get_attribute("href")
                if href and any(v in href for v in ["youtube.com", "youtu.be", "vimeo.com", "drive.google.com"]):
                    if href not in video_links:
                        video_links.append(href)
        except Exception:
            pass
        try:
            # Some cards embed the video player directly rather than linking out
            for tag, attr in [("video", "src"), ("source", "src"), ("iframe", "src")]:
                for el in card.find_elements(By.XPATH, f'.//{tag}'):
                    src = el.get_attribute(attr)
                    if src and src not in video_links and (
                        "youtube.com" in src or "vimeo.com" in src or src.endswith((".mp4", ".webm", ".mov"))
                    ):
                        video_links.append(src)
        except Exception:
            pass

        if video_links:
            text += "\n" + "\n".join(video_links)

        # 2. Extract image URL (skip logo/avatar, grab the actual post media image)
        image_url = None
        try:
            imgs = card.find_elements(By.XPATH, './/img')
            valid_imgs = []
            for img in imgs:
                src = img.get_attribute("src")
                if src and ("googleusercontent.com" in src or "gstatic.com" in src):
                    valid_imgs.append(src)
            image_url = pick_post_image(valid_imgs)
        except Exception:
            pass

        cta_text = None
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines and lines[-1] in CTA_LABELS:
            cta_text = lines[-1]

        content_hash = hashlib.md5(f"{competitor_name}|{text}".encode("utf-8")).hexdigest()

        return {
            "content": text,
            "cta_text": cta_text,
            "image_url": image_url,
            "post_url": None,
            "content_hash": content_hash,
        }
    except Exception:
        return None


def extract_posts_from_dialog(driver, competitor_name: str) -> List[Dict[str, Any]]:
    time.sleep(2)

    posts_by_hash: Dict[str, Dict[str, Any]] = {}

    def accumulate(_driver, dialog):
        """Called after every scroll step - the feed is virtualized, so we must
        grab whatever's currently rendered NOW, not just once at the end."""
        card_elements = _find_post_container(dialog)
        for card in card_elements:
            post = _extract_card(card, competitor_name)
            if post and post["content_hash"] not in posts_by_hash:
                posts_by_hash[post["content_hash"]] = post

    # Grab whatever's visible before we start scrolling too
    panel = _get_updates_dialog(driver)
    if panel is not None:
        accumulate(driver, panel)

    # Scroll fully to load all available updates, extracting at every step
    # so posts aren't lost when the virtualized list evicts earlier cards.
    scroll_modal_container(driver, on_scroll=accumulate)

    # One last pass in case anything rendered right after scrolling settled
    panel = _get_updates_dialog(driver)
    if panel is not None:
        accumulate(driver, panel)

    posts = list(posts_by_hash.values())

    if not posts:
        try:
            debug_path = f"debug_dialog_{competitor_name.replace(' ', '_')}.html"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(panel.get_attribute("outerHTML") if panel is not None else "<no panel found>")
            print(f"[!] No card elements found. Dumped panel HTML to {debug_path} for inspection.")
        except Exception as e:
            print(f"[!] No card elements found, and debug dump also failed: {e}")

        try:
            raw_text = panel.text if panel is not None else ""
            if raw_text:
                return parse_post_blocks(raw_text, competitor_name)
        except Exception:
            pass

    return posts


def parse_post_blocks(raw_text: str, competitor_name: str) -> List[Dict[str, Any]]:
    posts = []
    date_pattern = re.compile(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})")
    chunks = date_pattern.split(raw_text)

    if len(chunks) < 2:
        return []

    for i in range(1, len(chunks) - 1, 2):
        date_str = chunks[i].strip()
        body_text = chunks[i + 1].strip()

        full_content = f"{date_str}\n{body_text}".strip()
        if len(full_content) < 20:
            continue

        cta_text = None
        lines = full_content.split("\n")
        if lines[-1] in CTA_LABELS:
            cta_text = lines[-1]

        content_hash = hashlib.md5(f"{competitor_name}|{full_content}".encode("utf-8")).hexdigest()

        posts.append({
            "content": full_content,
            "cta_text": cta_text,
            "image_url": None,
            "post_url": None,
            "content_hash": content_hash,
        })

    return posts


def run_competitor_scrape(competitor_name: str, maps_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Public entry point main.py imports. Internally this now drives the
    test4.py-derived flow (google_search / open_updates / find_scroll_container
    / load_all_posts / scrape_post_urls) so you get the same detailed
    "what's happening / where are we stuck" print output test4.py gave you,
    while keeping this function's name, signature, and return shape
    (content/cta_text/image_url/post_url/content_hash) unchanged so nothing
    downstream breaks.
    """
    driver = init_driver()

    # Watchdog: if this competitor takes longer than COMPETITOR_WALL_CLOCK_BUDGET
    # for any reason (a stalled request that slips past the driver's own
    # timeouts, a wedged renderer, etc.), force-kill the browser from another
    # thread so the process can never sit there hung indefinitely. quit() is
    # safe to call twice / on an already-dead driver.
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

    try:
        print("\n" + "#" * 80)
        print(f"SCRAPING: {competitor_name}")
        print("#" * 80)
        section("GOOGLE SEARCH")
        print(f"[*] Query: {competitor_name}")    
        try:
            driver.get("https://www.google.com")
            print("[*] Checking for a cookie/consent dialog on homepage...")
            dismiss_consent_dialog(driver)
            print("[*] Typing query into the search bar...")
            search_box = WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_query = competitor_name + " " + str(maps_url)
            search_box.clear()  
            search_box.send_keys(search_query)
            search_box.send_keys(Keys.RETURN) 
                    
        except TimeoutException:
            print(f"[-] {competitor_name}: Search bar interaction timed out after {PAGE_LOAD_TIMEOUT}s. Skipping.")
            return []
                    
        print("[*] Search results loaded, waiting for the page to settle...")
        time.sleep(3)
        
        print("[*] Checking for CAPTCHA / bot-detection...")
        try:
            page_is_captcha(driver)
        except CaptchaBlocked:
            print(f"[-] {competitor_name}: CAPTCHA not solved in time. Skipping.")
            return []

        # --- 2. Open the "Updates" popup (test4's find_previous_updates + open_updates) ---
        section("LOOKING FOR PREVIOUS UPDATES")
        try:
            popup = open_updates(driver)
        except RuntimeError as e:
            print(f"[-] {competitor_name}: {e} Skipping.")
            return []

        try:
            page_is_captcha(driver)
        except CaptchaBlocked:
            print(f"[-] {competitor_name}: CAPTCHA not solved in time. Skipping.")
            return []

        # --- 3. Find the scrollable feed inside the popup ---
        try:
            scroll_container = find_scroll_container(driver, popup)
        except RuntimeError as e:
            print(f"[-] {competitor_name}: {e} Skipping.")
            return []

        # --- 4. Scroll and load every post card (test4's load_all_posts) ---
        raw_posts = load_all_posts(
            driver, scroll_container,
            max_scrolls=UPDATES_MAX_SCROLLS,
            scroll_pause=UPDATES_SCROLL_PAUSE,
            max_posts=UPDATES_MAX_POSTS,
        )
        if not raw_posts:
            print(f"[-] {competitor_name}: no update cards found.")
            return []

        section("CARD PREVIEW")
        for i, p in enumerate(raw_posts, 1):
            preview = p.get("content", "")[:180].replace("\n", " ")
            print(f"[*] {i:02d}. {p.get('date', '')} | {preview}")

        # --- 5. Visit each card's Share popup to get its post URL (test4's scrape_post_urls) ---
        raw_posts = scrape_post_urls(driver, scroll_container, raw_posts, max_scrolls=UPDATES_MAX_SCROLLS)

        # --- 6. Translate into this module's existing post shape ---
        section("FINALIZING POSTS")
        posts = []
        for p in raw_posts:
            content = p.get("content", "")

            image_url = (p.get("imageUrl") or "").strip() or None
            video_url = (p.get("videoUrl") or "").strip() or None
            # Mutually exclusive by design: image takes priority (a post
            # rarely has both); video_url is only populated when there's no
            # image. Neither gets stuffed into `content` anymore - they're
            # their own fields so there's no ambiguity about which is which.
            if image_url:
                video_url = None

            lines = [line.strip() for line in content.split("\n") if line.strip()]
            cta_text = lines[-1] if lines and lines[-1] in CTA_LABELS else None

            content_hash = hashlib.md5(f"{competitor_name}|{content}".encode("utf-8")).hexdigest()

            posts.append({
                "date": p.get("date", ""),
                "content": content,
                "cta_text": cta_text,
                "image_url": image_url,
                "video_url": video_url,
                "post_url": (p.get("post_url") or "").strip() or None,
                "content_hash": content_hash,
            })

        print(f"[+] Successfully collected {len(posts)} posts for {competitor_name}.")
        return posts

    except WebDriverException as e:
        # The watchdog killing the driver mid-call surfaces as one of these -
        # treat it as "skip and move on" rather than letting it propagate and
        # take the whole batch down.
        if watchdog_fired.is_set():
            print(f"[-] {competitor_name}: stopped by watchdog timeout.")
        else:
            print(f"[-] {competitor_name}: browser/session error ({e.__class__.__name__}). Skipping.")
        return []
    finally:
        timer.cancel()
        try:
            driver.quit()
        except Exception:
            pass


# --- Tunable defaults (all overridable per-call via scrape_business_updates) ---
UPDATES_HEADLESS = False
UPDATES_WAIT_TIMEOUT = 30
UPDATES_MAX_SCROLLS = 60
UPDATES_SCROLL_PAUSE = 1.25
UPDATES_MAX_POSTS = 100

UPDATES_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def clean_text(value):
    if not value:
        return ""
    value = str(value).replace("\r", "\n").replace("\r\n", "\n")
    lines = []
    for line in value.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def norm(value):
    return re.sub(r"\s+", " ", clean_text(value).lower()).strip()


def create_driver(headless=None):
    options = Options()
    if headless is None:
        headless = UPDATES_HEADLESS
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-notifications")
    options.add_argument("--lang=en-US")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    try:
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
    except Exception:
        pass
    return driver


def captcha_present(driver):
    selectors = [
        "iframe[src*='recaptcha']",
        "iframe[title*='reCAPTCHA']",
        ".g-recaptcha",
        "#captcha-form",
        "[data-sitekey]",
    ]
    for selector in selectors:
        try:
            if any(e.is_displayed() for e in driver.find_elements(By.CSS_SELECTOR, selector)):
                return True
        except Exception:
            pass
    try:
        text = driver.find_element(By.TAG_NAME, "body").text.lower()
    except Exception:
        text = ""
    phrases = (
        "our systems have detected unusual traffic",
        "unusual traffic from your computer network",
        "verify you are human",
        "confirm you are not a robot",
        "i'm not a robot",
        "please verify that you're not a robot",
    )
    return any(p in text for p in phrases)


def wait_for_captcha(driver):
    if not captcha_present(driver):
        return
    section("CAPTCHA DETECTED")
    print("Solve the CAPTCHA manually in Chrome.")
    print("The scraper will wait until the CAPTCHA disappears.")
    print("=" * 80)
    while captcha_present(driver):
        print("Waiting for manual CAPTCHA solution...", flush=True)
        time.sleep(3)
    print("CAPTCHA solved. Continuing...")
    time.sleep(2)


def google_search(driver, query):
    section("GOOGLE SEARCH")
    print(f"Query: {query}")
    driver.get("https://www.google.com/search?q=" + quote_plus(query))
    time.sleep(3)
    wait_for_captcha(driver)


def find_previous_updates(driver):
    """Locate the Google Business Profile 'View previous updates on Google' link.

    Google frequently renders the visible link as a span/div in the right-hand
    Knowledge Panel.  The reliable anchor is the section heading immediately
    above it, whose text starts with 'Updates from ...'.  We therefore locate
    that heading first, restrict the search to its nearby/right-panel
    container, and then find the View-previous-updates text inside it.
    """
    phrase = "view previous updates on google"
    heading_phrase = "updates from"

    def visible(el):
        try:
            return el.is_displayed() and el.size.get("width", 0) > 0 and el.size.get("height", 0) > 0
        except Exception:
            return False

    def click_element(el):
        """Return True if an actual click sequence was dispatched."""
        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({block:'center',inline:'center'});",
                el,
            )
            time.sleep(0.5)
        except Exception:
            pass

        # Normal Selenium click.
        try:
            el.click()
            return True
        except Exception:
            pass

        # JS click on the exact element.
        try:
            driver.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            pass

        # Dispatch the mouse sequence because some Google elements listen for
        # pointer/mouse events instead of a simple HTMLElement.click().
        try:
            driver.execute_script(r"""
                const el = arguments[0];
                ['pointerover','mouseover','pointerdown','mousedown','pointerup','mouseup','click']
                  .forEach(type => el.dispatchEvent(new MouseEvent(type, {
                      bubbles: true, cancelable: true, view: window, buttons: 1
                  })));
            """, el)
            return True
        except Exception:
            pass

        try:
            ActionChains(driver).move_to_element(el).pause(0.3).click().perform()
            return True
        except Exception:
            return False

    for attempt in range(30):
        try:
            # -----------------------------------------------------------------
            # 1. Find the right-panel section heading: "Updates from ..."
            # -----------------------------------------------------------------
            headings = driver.find_elements(
                By.XPATH,
                "//*[contains(translate(normalize-space(.), "
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), "
                "'updates from')]"
            )

            heading_candidates = []
            for h in headings:
                try:
                    if not visible(h):
                        continue
                    text = norm(h.text).lower()
                    if "updates from" not in text:
                        continue
                    # Avoid giant page-level ancestors containing many unrelated
                    # occurrences. Prefer compact heading-like elements.
                    if len(text) > 250:
                        continue
                    heading_candidates.append(h)
                except Exception:
                    continue

            heading_candidates.sort(key=lambda x: len(norm(x.text)))

            for heading in heading_candidates:
                try:
                    # Search progressively larger ancestors around this heading.
                    # The link is normally a sibling/child of the heading in the
                    # same Knowledge Panel section.
                    containers = []
                    cur = heading
                    for _ in range(6):
                        if cur is None:
                            break
                        containers.append(cur)
                        try:
                            cur = cur.find_element(By.XPATH, "..")
                        except Exception:
                            break

                    for container in containers:
                        try:
                            # First search descendants for the exact visible text.
                            matches = container.find_elements(
                                By.XPATH,
                                ".//*[contains(translate(normalize-space(.), "
                                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), "
                                f"'{phrase}')]"
                            )
                            matches = [m for m in matches if visible(m)]
                            matches.sort(key=lambda x: len(norm(x.text)))

                            if not matches:
                                continue

                            # Prefer a real link/button/role element.
                            semantic = []
                            for m in matches:
                                try:
                                    tag = m.tag_name.lower()
                                    role = (m.get_attribute("role") or "").lower()
                                    href = m.get_attribute("href")
                                    if tag in ("a", "button") or role in ("link", "button") or href:
                                        semantic.append(m)
                                except Exception:
                                    pass

                            target = (semantic or matches)[0]
                            print("Found 'Updates from ...' section in the right panel.")
                            print("Found 'View previous updates on Google'.")
                            print("Opening 'View previous updates on Google'...")
                            if click_element(target):
                                return target
                        except Exception:
                            continue
                except Exception:
                    continue

            # -----------------------------------------------------------------
            # 2. Fallback: exact text anywhere on the page, then walk upward.
            # -----------------------------------------------------------------
            exact = driver.find_elements(
                By.XPATH,
                "//*[contains(translate(normalize-space(.), "
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'), "
                f"'{phrase}')]"
            )
            exact = [e for e in exact if visible(e)]
            exact.sort(key=lambda x: len(norm(x.text)))

            for e in exact:
                try:
                    # Walk up only a few levels. Google normally puts the click
                    # listener on one of these immediate ancestors.
                    candidates = [e]
                    cur = e
                    for _ in range(5):
                        try:
                            cur = cur.find_element(By.XPATH, "..")
                            candidates.append(cur)
                        except Exception:
                            break
                    for candidate in candidates:
                        if not visible(candidate):
                            continue
                        if click_element(candidate):
                            return candidate
                except Exception:
                    continue

        except Exception:
            pass

        time.sleep(1)

    return None

def open_updates(driver):
    section("LOOKING FOR PREVIOUS UPDATES")
    button = find_previous_updates(driver)
    if button is None:
        raise RuntimeError("Could not find 'View previous updates on Google'.")

    print("Found 'View previous updates on Google'.")
    print("Opening 'View previous updates on Google'...")

    # Scroll the actual clickable element into view first.
    try:
        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center', inline:'center'});",
            button,
        )
        time.sleep(0.8)
    except Exception:
        pass

    # Google can attach the click handler to a parent while the visible text
    # sits inside a span. Try a normal Selenium click first, then JS click.
    clicked = False
    try:
        button.click()
        clicked = True
    except (ElementClickInterceptedException, WebDriverException):
        pass

    if not clicked:
        try:
            driver.execute_script("arguments[0].click();", button)
            clicked = True
        except Exception:
            pass

    # Last-resort click: dispatch a real mouse event from the element's center.
    if not clicked:
        try:
            ActionChains(driver).move_to_element(button).pause(0.2).click().perform()
            clicked = True
        except Exception:
            pass

    if not clicked:
        raise RuntimeError("Found 'View previous updates on Google' but could not click it.")

    section("WAITING FOR UPDATES POPUP")

    def popup_ready(d):
        dialogs = d.find_elements(By.CSS_SELECTOR, "[role='dialog'], [aria-modal='true']")
        for dialog in dialogs:
            try:
                if dialog.is_displayed():
                    text = norm(dialog.text)
                    # The Updates dialog normally contains 'Updates' and/or
                    # 'Recent updates'. This avoids accepting unrelated dialogs.
                    if 'recent updates' in text or text.startswith('updates') or '\nupdates\n' in text:
                        return dialog
            except Exception:
                pass
        return False

    try:
        popup = WebDriverWait(driver, UPDATES_WAIT_TIMEOUT).until(popup_ready)
    except TimeoutException:
        # A few Google layouts do not expose role=dialog reliably. Fall back
        # to any visible large dialog only after the click has already happened.
        def any_large_dialog(d):
            dialogs = d.find_elements(By.CSS_SELECTOR, "[role='dialog'], [aria-modal='true']")
            for dialog in dialogs:
                try:
                    if dialog.is_displayed() and len(dialog.text.strip()) > 20:
                        return dialog
                except Exception:
                    pass
            return False
        popup = WebDriverWait(driver, UPDATES_WAIT_TIMEOUT).until(any_large_dialog)

    print("Updates popup detected.")
    time.sleep(1.5)
    return popup


def find_scroll_container(driver, popup, retries=10, wait_seconds=1.0):
    section("FINDING POPUP SCROLL AREA")
    script = """
    const root = arguments[0];
    const all = [root, ...root.querySelectorAll('*')];
    const candidates = [];
    for (const el of all) {
        try {
            const s = getComputedStyle(el);
            const overflow = ['auto','scroll','overlay'].includes(s.overflowY);
            const distance = el.scrollHeight - el.clientHeight;
            if (overflow && distance > 100 && el.clientHeight > 200) {
                candidates.push({el, distance, area: el.clientWidth * el.clientHeight});
            }
        } catch(e) {}
    }
    candidates.sort((a,b) => b.distance - a.distance || b.area - a.area);
    return candidates.length ? candidates[0].el : null;
    """
    # The popup can take a few seconds to actually render its content after
    # it appears (spinner -> cards), so a scrollable area with real height
    # may not exist yet on the first check. Poll instead of failing instantly.
    container = None
    for attempt in range(1, retries + 1):
        container = driver.execute_script(script, popup)
        if container is not None:
            break
        print(f"[*] Popup still loading, no scrollable content yet (check {attempt}/{retries})...")
        time.sleep(wait_seconds)

    if container is None:
        raise RuntimeError("Could not find the Updates popup scroll container.")
    print("Scrollable area found.")
    return container


def extract_visible_cards(driver, scroll_container):
    """
    Find likely card containers from DATE LEAVES, but require a Share control
    or meaningful media/content. This prevents every nested date/text element
    from becoming a separate post.
    """
    script = """
    const root = arguments[0];
    const dateRe = /\\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\\s+\\d{1,2},?\\s+\\d{4}\\b/i;

    function visible(el) {
        try {
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
        } catch(e) { return false; }
    }

    function hasShare(el) {
        const nodes = el.querySelectorAll('button,[role="button"],[aria-label],[title],[data-tooltip]');
        for (const n of nodes) {
            const label = ((n.getAttribute('aria-label') || '') + ' ' + (n.getAttribute('title') || '') + ' ' + (n.getAttribute('data-tooltip') || '') + ' ' + (n.innerText || '')).toLowerCase();
            if (label.includes('share')) return true;
        }
        return false;
    }

    const leaves = [];
    for (const el of root.querySelectorAll('*')) {
        try {
            if (el.children.length !== 0 || !visible(el)) continue;
            const t = (el.innerText || el.textContent || '').trim();
            if (t && dateRe.test(t)) leaves.push(el);
        } catch(e) {}
    }

    const results = [];
    const seen = new Set();

    for (const leaf of leaves) {
        let cur = leaf;
        let best = null;
        for (let level = 0; level < 14 && cur && cur !== root; level++, cur = cur.parentElement) {
            try {
                if (!visible(cur)) continue;
                const text = (cur.innerText || '').trim();
                const dates = text.match(new RegExp(dateRe.source, 'ig')) || [];
                const r = cur.getBoundingClientRect();
                const imgs = cur.querySelectorAll('img').length;
                const videos = cur.querySelectorAll('video,source').length;
                const share = hasShare(cur);
                if (dates.length === 1 && text.length >= 30 && text.length <= 5000 && r.width >= 250 && r.height >= 80 && (share || imgs || videos)) {
                    best = cur;
                }
                if (share && dates.length === 1 && text.length >= 30 && r.height >= 100) {
                    best = cur;
                    break;
                }
            } catch(e) {}
        }
        if (best) {
            const text = (best.innerText || '').trim();
            const key = text.slice(0, 1200);
            if (!seen.has(key)) {
                seen.add(key);
                results.push(best);
            }
        }
    }
    return results;
    """
    try:
        return driver.execute_script(script, scroll_container) or []
    except Exception:
        return []


_STILL_TRUNCATED_RE = re.compile(r'(\.\.\.|\u2026)?\s*more\s*$', re.IGNORECASE)


def _card_is_truncated(driver, card):
    try:
        text = driver.execute_script("return (arguments[0].innerText || '').trim();", card)
    except Exception:
        return False
    # Only look at the tail - "more" appearing earlier in real content
    # (e.g. "...and more!") shouldn't count as truncation.
    return bool(_STILL_TRUNCATED_RE.search((text or '')[-40:]))


def expand_card_content(driver, card, max_attempts=3):
    """
    Google truncates long post text and appends a small 'More' toggle,
    often rendered as '\u2026 More' glued onto the last line rather than as
    a cleanly isolated 'More' string - so an exact text === 'more' match
    misses it. Find every short, on-screen element whose text contains
    'more', click through candidates until the truncation marker is
    actually gone, and give up gracefully if nothing works (e.g. the card
    genuinely doesn't have a toggle).
    """
    if not _card_is_truncated(driver, card):
        return False

    find_script = r"""
    const card = arguments[0];
    const nodes = [...card.querySelectorAll('button,[role="button"],span,div,a')];
    const matches = [];
    for (const n of nodes) {
        try {
            const t = (n.innerText || '').trim();
            if (!t || t.length > 15) continue;
            if (!/more/i.test(t)) continue;
            const r = n.getBoundingClientRect();
            if (r.width <= 0 || r.height <= 0) continue;
            matches.push(n);
        } catch(e) {}
    }
    return matches;
    """
    try:
        candidates = driver.execute_script(find_script, card) or []
    except Exception:
        candidates = []

    for n in candidates[:max_attempts]:
        try:
            driver.execute_script(
                "try { arguments[0].click(); } "
                "catch(e) { arguments[0].dispatchEvent(new MouseEvent('click', "
                "{bubbles:true, cancelable:true, view:window})); }",
                n,
            )
        except Exception:
            continue
        time.sleep(0.4)
        if not _card_is_truncated(driver, card):
            return True

    return False


def _try_resolve_video(driver, card):
    """
    Some post cards show a video behind a thumbnail + play-button overlay -
    the real <video>/<iframe> src doesn't exist in the DOM until playback is
    triggered. Click the play control (if one is found) and see if a
    resolvable video/embed source appears afterward. Best-effort: returns
    '' if nothing playable could be found.
    """
    find_play_button = r"""
    const card = arguments[0];
    const nodes = card.querySelectorAll('[aria-label], [title], button, [role="button"]');
    for (const n of nodes) {
        try {
            const label = ((n.getAttribute('aria-label') || '') + ' ' + (n.getAttribute('title') || '')).toLowerCase();
            if (!label.includes('play')) continue;
            const r = n.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) return n;
        } catch(e) {}
    }
    return null;
    """
    read_video_src = r"""
    const card = arguments[0];
    for (const v of card.querySelectorAll('video,video source,source')) {
        const src = v.currentSrc || v.src || v.getAttribute('src') || '';
        if (src && !src.startsWith('blob:') && !src.startsWith('data:')) return src;
    }
    for (const f of card.querySelectorAll('iframe')) {
        const src = f.src || f.getAttribute('src') || '';
        if (src && (src.includes('youtube') || src.includes('vimeo'))) return src;
    }
    return '';
    """
    try:
        btn = driver.execute_script(find_play_button, card)
        if not btn:
            return ''
        driver.execute_script(
            "try { arguments[0].click(); } "
            "catch(e) { arguments[0].dispatchEvent(new MouseEvent('click', "
            "{bubbles:true, cancelable:true, view:window})); }",
            btn,
        )
        time.sleep(0.9)
        return driver.execute_script(read_video_src, card) or ''
    except Exception:
        return ''


def extract_card_data(driver, card):
    # Expand truncated content ("...More") before reading the card's text,
    # so `content` below is the full post rather than the truncated preview.
    expand_card_content(driver, card)

    script = """
    const card = arguments[0];
    const dateRe = /\\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\\s+\\d{1,2},?\\s+\\d{4}\\b/i;
    const lines = (card.innerText || '').split(/\\n+/).map(x => x.trim()).filter(Boolean);

    let date = '';
    let dateIndex = -1;
    for (let i=0; i<lines.length; i++) {
        const m = lines[i].match(dateRe);
        if (m) { date = m[0]; dateIndex = i; break; }
    }
    // NOTE: some cards (event/promo posts like "X Fest, 18th Nov - 10th Dec")
    // don't render an absolute "Mon D, YYYY" posted-date at all - only a
    // date *range* inside the body copy, in a format the regex above won't
    // match. Previously we dropped the whole card when no date line was
    // found; now we just leave `date` blank and keep the content/media -
    // losing the post entirely is worse than a blank date column.
    const businessName = dateIndex > 0 ? lines[dateIndex - 1] : '';
    const contentLines = dateIndex >= 0 ? lines.slice(dateIndex + 1) : lines;
    let content = contentLines
        .filter(x => !['more','less','svg'].includes(x.toLowerCase()))
        .join('\\n');
    // Safety net: if a click-to-expand attempt still left a trailing
    // "...More" / "\u2026 More" fragment glued onto the last line, strip it
    // rather than keep it as literal scraped text.
    content = content.replace(/(\\.\\.\\.|\\u2026)?\\s*more\\s*$/i, '').trimEnd();

    const businessLower = businessName.toLowerCase().trim();
    const images = [];
    for (const img of card.querySelectorAll('img')) {
        try {
            const src = img.currentSrc || img.src || img.getAttribute('src') || img.getAttribute('data-src') || img.getAttribute('data-lazy-src') || '';
            if (!src || src.startsWith('data:')) continue;
            const alt = (img.getAttribute('alt') || '').trim();
            const w = img.naturalWidth || img.width || 0;
            const h = img.naturalHeight || img.height || 0;
            const area = w * h;
            if (businessLower && alt.toLowerCase().includes(businessLower)) continue;
            images.push({src, alt, area, w, h});
        } catch(e) {}
    }
    images.sort((a,b) => b.area - a.area);
    const imageUrl = images.length ? images[0].src : '';

    let videoUrl = '';
    for (const v of card.querySelectorAll('video,video source,source')) {
        try {
            const src = v.currentSrc || v.src || v.getAttribute('src') || '';
            if (src && !src.startsWith('blob:') && !src.startsWith('data:')) { videoUrl = src; break; }
        } catch(e) {}
    }

    // A video post is often just a thumbnail <img> (picked up above as
    // imageUrl) with a play-button overlay, and no actual <video> tag until
    // you click play. Flag that case so Python can attempt to resolve the
    // real source - otherwise a video post silently gets miscounted as a
    // plain image post.
    let hasPlayIndicator = false;
    if (!videoUrl) {
        const markers = card.querySelectorAll('[aria-label], [title]');
        for (const n of markers) {
            try {
                const label = ((n.getAttribute('aria-label') || '') + ' ' + (n.getAttribute('title') || '')).toLowerCase();
                if (label.includes('play')) { hasPlayIndicator = true; break; }
            } catch(e) {}
        }
    }

    const signature = date + '||' + content.slice(0, 1000);
    return {date, businessName, content, imageUrl, videoUrl, hasPlayIndicator, signature};
    """
    try:
        data = driver.execute_script(script, card)
    except Exception:
        return None

    if not data:
        return None

    # Best-effort resolution of a video hidden behind a play-button overlay.
    if not data.get("videoUrl") and data.get("hasPlayIndicator"):
        resolved = _try_resolve_video(driver, card)
        if resolved:
            data["videoUrl"] = resolved
            data["imageUrl"] = ""  # it's a video post, not an image post
        # If unresolved, we deliberately leave imageUrl as-is (the thumbnail)
        # rather than mislabeling a non-playable thumbnail URL as a video.

    return data


def load_all_posts(driver, scroll_container, max_scrolls=None, scroll_pause=None, max_posts=None):
    max_scrolls = UPDATES_MAX_SCROLLS if max_scrolls is None else max_scrolls
    scroll_pause = UPDATES_SCROLL_PAUSE if scroll_pause is None else scroll_pause
    max_posts = UPDATES_MAX_POSTS if max_posts is None else max_posts

    section("LOADING ALL AVAILABLE UPDATES")
    posts = {}
    last_height = -1
    stable_bottom = 0

    # The popup can sit on a loading spinner for a few seconds before any
    # card actually renders. If we start the scroll/stability logic while
    # it's still empty, scrollHeight <= clientHeight looks exactly like
    # "reached the end", so we'd stop immediately with 0 posts. Wait for at
    # least one real card first.
    print("[*] Waiting for the first update card to render...")
    first_card_deadline = time.time() + 20
    while time.time() < first_card_deadline:
        if extract_visible_cards(driver, scroll_container):
            print("[*] First card detected, starting scroll pass.")
            break
        time.sleep(1)
    else:
        print("[*] Still nothing rendered after 20s - proceeding anyway in case this business has no updates.")

    for i in range(1, max_scrolls + 1):
        cards = extract_visible_cards(driver, scroll_container)
        new_count = 0
        for card in cards:
            try:
                data = extract_card_data(driver, card)
                if not data or not data.get('date') or not data.get('content'):
                    continue
                key = norm(data['date']) + '||' + norm(data['content'][:1000])
                if key not in posts:
                    posts[key] = data
                    new_count += 1
            except (StaleElementReferenceException, WebDriverException):
                continue

        state = driver.execute_script("""
            const e = arguments[0];
            return {top:e.scrollTop, height:e.scrollHeight, client:e.clientHeight};
        """, scroll_container)
        top, height, client = state['top'], state['height'], state['client']
        at_bottom = top + client >= height - 10
        print(f"Scroll {i:02d} | top={int(top)} | height={int(height)} | cards={len(posts)} | new={new_count}")

        if at_bottom and not posts:
            # Bottom of an empty/not-yet-loaded container isn't "done" -
            # give it more time instead of stopping.
            stable_bottom = 0
        elif at_bottom:
            stable_bottom += 1
        else:
            stable_bottom = 0
        if stable_bottom >= 3:
            print("Reached end of updates.")
            break

        driver.execute_script("""
            const e = arguments[0];
            e.scrollTop = Math.min(e.scrollTop + Math.floor(e.clientHeight * 0.80), e.scrollHeight);
        """, scroll_container)
        time.sleep(scroll_pause)

        if height == last_height and at_bottom and posts:
            stable_bottom += 1
        last_height = height

        if len(posts) >= max_posts:
            print(f"Reached safety limit of {max_posts} posts.")
            break

    section("IDENTIFYING INDIVIDUAL UPDATE CARDS")
    print(f"Unique update cards found: {len(posts)}")
    return list(posts.values())


def find_card_by_signature(driver, scroll_container, signature):
    """
    Re-locate the exact card whose extract_card_data() signature matches
    `signature`. This reuses extract_visible_cards/extract_card_data (which
    now expand any 'More' truncation before reading text) instead of a
    separate inline JS text-matcher, so a card's signature is computed the
    same way here as it was when it was first collected in load_all_posts -
    otherwise a still-truncated re-read would never match the expanded
    signature captured earlier.
    """
    try:
        cards = extract_visible_cards(driver, scroll_container)
    except Exception:
        return None

    for card in cards:
        try:
            data = extract_card_data(driver, card)
            if data and data.get('signature') == signature:
                return card
        except (StaleElementReferenceException, WebDriverException):
            continue
        except Exception:
            continue
    return None


def find_share_button(driver, card):
    script = """
    const card = arguments[0];
    const nodes = card.querySelectorAll('button,[role="button"],[aria-label],[title],[data-tooltip]');
    for (const n of nodes) {
        try {
            const label = ((n.getAttribute('aria-label')||'')+' '+(n.getAttribute('title')||'')+' '+(n.getAttribute('data-tooltip')||'')+' '+(n.innerText||'')).toLowerCase();
            const r=n.getBoundingClientRect();
            if(label.includes('share') && r.width>0 && r.height>0) return n;
        } catch(e) {}
    }
    return null;
    """
    try:
        return driver.execute_script(script, card)
    except Exception:
        return None


def share_popup_visible(driver):
    try:
        text = driver.find_element(By.TAG_NAME, 'body').text.lower()
        return 'click to copy link' in text or ('copy link' in text and 'share' in text)
    except Exception:
        return False


def extract_share_url(driver):
    """Extract the post URL from Google's Share popup.

    Google currently shows the post URL underneath "Click to copy link".
    We deliberately inspect the *share dialog only* so URLs from the
    underlying Updates popup are never mistaken for the post URL.
    """
    script = r"""
    function visible(el) {
        try {
            const r = el.getBoundingClientRect();
            const s = getComputedStyle(el);
            return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
        } catch(e) { return false; }
    }

    function clean(u) {
        return (u || '').replace(/[\\),.;]+$/, '').trim();
    }

    function findUrl(text) {
        const matches = String(text || '').match(/https?:\/\/[^\s<>\"']+/g) || [];
        for (const m of matches) {
            const u = clean(m);
            if (/^https?:\/\/(?:share\.google|www\.google|google\.)/i.test(u)) return u;
            if (/^https?:\/\//i.test(u)) return u;
        }
        return '';
    }

    const dialogs = [...document.querySelectorAll('[role="dialog"], [aria-modal="true"]')]
        .filter(visible);

    // The last visible matching dialog is normally the Share dialog.
    for (const dialog of dialogs.reverse()) {
        const text = dialog.innerText || '';
        const lower = text.toLowerCase();
        if (!lower.includes('click to copy link') &&
            !lower.includes('copy link') &&
            !lower.includes('share')) continue;

        // 1. Inputs / textareas.
        for (const el of dialog.querySelectorAll('input, textarea')) {
            if (!visible(el)) continue;
            const v = el.value || el.getAttribute('value') || '';
            if (/^https?:\/\//i.test(v)) return clean(v);
        }

        // 2. Explicit hrefs.
        for (const a of dialog.querySelectorAll('a[href]')) {
            if (!visible(a)) continue;
            const href = a.href || a.getAttribute('href') || '';
            if (/^https?:\/\//i.test(href)) return clean(href);
        }

        // 3. The URL displayed as text below "Click to copy link".
        const url = findUrl(text);
        if (url) return url;

        // 4. Inspect small text elements around "Click to copy link".
        const nodes = [...dialog.querySelectorAll('*')];
        for (const el of nodes) {
            if (!visible(el)) continue;
            const t = (el.innerText || '').trim().toLowerCase();
            if (t !== 'click to copy link' && t !== 'click to copy') continue;
            let cur = el;
            for (let i = 0; i < 5 && cur; i++, cur = cur.parentElement) {
                const u = findUrl(cur.innerText || '');
                if (u) return u;
            }
        }
    }
    return '';
    """
    try:
        return driver.execute_script(script) or ''
    except Exception:
        return ''


def copy_share_url_from_popup(driver):
    """Click Google's 'Click to copy link' control and read the clipboard.

    This is the primary method because the URL shown in Google's Share
    dialog is sometimes rendered as non-link text and is not exposed as an
    input or anchor. The clipboard is granted to the current Google origin
    through Chrome DevTools before reading it.
    """
    try:
        # Grant clipboard access for the current origin where supported.
        try:
            origin = driver.execute_script("return location.origin;")
            driver.execute_cdp_cmd(
                "Browser.grantPermissions",
                {
                    "origin": origin,
                    "permissions": ["clipboardReadWrite", "clipboardSanitizedWrite"],
                },
            )
        except Exception:
            pass

        result = driver.execute_async_script(r"""
            const done = arguments[arguments.length - 1];
            function visible(el) {
                try {
                    const r = el.getBoundingClientRect();
                    const s = getComputedStyle(el);
                    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
                } catch(e) { return false; }
            }
            const dialogs = [...document.querySelectorAll('[role="dialog"], [aria-modal="true"]')]
                .filter(visible);
            let target = null;
            for (const dialog of dialogs.reverse()) {
                const text = (dialog.innerText || '').toLowerCase();
                if (!text.includes('copy link') && !text.includes('click to copy')) continue;
                const nodes = [...dialog.querySelectorAll('button,[role="button"],a,[tabindex],div,span')];
                for (const n of nodes) {
                    if (!visible(n)) continue;
                    const t = (n.innerText || '').trim().toLowerCase();
                    const a = ((n.getAttribute('aria-label') || '') + ' ' +
                               (n.getAttribute('title') || '')).trim().toLowerCase();
                    if (t === 'click to copy link' || t === 'click to copy' ||
                        a.includes('click to copy') || a === 'copy link') {
                        target = n;
                        break;
                    }
                }
                if (target) break;
            }
            if (!target) { done(''); return; }
            try { target.click(); } catch(e) {
                try { target.dispatchEvent(new MouseEvent('click', {bubbles:true,cancelable:true,view:window})); }
                catch(e2) {}
            }
            setTimeout(async () => {
                try {
                    const text = await navigator.clipboard.readText();
                    done(text || '');
                } catch(e) {
                    done('');
                }
            }, 500);
        """)
        if result and re.match(r'^https?://', result.strip(), re.I):
            return result.strip()
    except Exception:
        pass
    return ''


def share_popup_open(driver):
    """Return True only when a visible Share/Copy Link popup exists."""
    try:
        return bool(driver.execute_script(r"""
            function visible(el) {
                try {
                    const r = el.getBoundingClientRect();
                    return r.width > 0 && r.height > 0;
                } catch(e) { return false; }
            }
            const dialogs = [...document.querySelectorAll('[role="dialog"], [aria-modal="true"]')];
            return dialogs.some(d => {
                if (!visible(d)) return false;
                const t = (d.innerText || '').toLowerCase();
                return t.includes('copy link') || t.includes('click to copy') || t.includes('share');
            });
        """))
    except Exception:
        return False


def close_share_popup(driver):
    """Close the Share popup without intentionally closing Updates."""
    # Prefer a close button inside the visible share dialog.
    try:
        buttons = driver.execute_script(r"""
            function visible(el) {
                try {
                    const r = el.getBoundingClientRect();
                    return r.width > 0 && r.height > 0;
                } catch(e) { return false; }
            }
            const dialogs = [...document.querySelectorAll('[role="dialog"], [aria-modal="true"]')]
                .filter(visible);
            for (const d of dialogs.reverse()) {
                const t = (d.innerText || '').toLowerCase();
                if (!t.includes('copy link') && !t.includes('click to copy') && !t.includes('share')) continue;
                const nodes = [...d.querySelectorAll('button,[role="button"],[aria-label],[title]')];
                for (const n of nodes) {
                    const label = ((n.getAttribute('aria-label') || '') + ' ' +
                                   (n.getAttribute('title') || '') + ' ' +
                                   (n.innerText || '')).toLowerCase().trim();
                    if (label === 'close' || label.startsWith('close ')) return n;
                }
            }
            return null;
        """)
        if buttons:
            try:
                buttons.click()
            except Exception:
                driver.execute_script('arguments[0].click();', buttons)
            time.sleep(0.5)
            return
    except Exception:
        pass

    # Escape is the fallback. If Google closes the Share dialog first,
    # leave the Updates popup open.
    try:
        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
        time.sleep(0.5)
    except Exception:
        pass


def get_post_url(driver, scroll_container, signature):
    """Find one exact card, open its Share popup, and get its post URL."""
    card = find_card_by_signature(driver, scroll_container, signature)
    if card is None:
        return ''

    try:
        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center',inline:'center'});",
            card,
        )
        time.sleep(0.7)
    except Exception:
        pass

    button = find_share_button(driver, card)
    if button is None:
        return ''

    if share_popup_open(driver):
        close_share_popup(driver)
        time.sleep(0.6)

    try:
        button.click()
    except Exception:
        try:
            driver.execute_script('arguments[0].click();', button)
        except Exception:
            return ''

    # Wait for the Share popup itself, then first use the visible URL and
    # finally use Google's built-in "Click to copy link" mechanism.
    for _ in range(30):
        time.sleep(0.25)
        if share_popup_open(driver):
            break

    url = extract_share_url(driver)
    if not url:
        url = copy_share_url_from_popup(driver)

    # One more DOM attempt after the clipboard click because Google can
    # update the dialog asynchronously.
    if not url:
        time.sleep(0.5)
        url = extract_share_url(driver)

    close_share_popup(driver)
    time.sleep(0.7)
    return url.strip()


def scrape_post_urls(driver, scroll_container, posts, max_scrolls=None):
    """
    Get the Share URL for EVERY discovered post.

    Important change from the previous implementation:
    instead of processing post #1, then jumping around the popup for
    post #2, #3, #4..., we walk through the Updates scroll container
    from top to bottom and process cards while they are actually in
    the DOM. This is much more reliable with Google's lazy-loaded
    update cards and prevents only the first Share URL being captured.
    """
    max_scrolls = UPDATES_MAX_SCROLLS if max_scrolls is None else max_scrolls
    section("GETTING POST LINKS FROM EACH CARD")

    # Map the exact signature to the Python post object.
    target = {p['signature']: p for p in posts}
    processed = set()

    # Start from the top.
    try:
        driver.execute_script(
            "arguments[0].scrollTop = 0;",
            scroll_container,
        )
        time.sleep(1.5)
    except Exception:
        pass

    stable_bottom = 0
    previous_state = None

    for scroll_no in range(1, max_scrolls + 1):
        # Find the cards currently rendered by Google.
        cards = extract_visible_cards(driver, scroll_container)

        for card in cards:
            try:
                data = extract_card_data(driver, card)
                if not data:
                    continue

                signature = data.get('signature', '')
                if signature not in target or signature in processed:
                    continue

                post = target[signature]
                number = posts.index(post) + 1

                print(
                    f"[{number:02d}/{len(posts):02d}] "
                    f"{post['date']} | getting Share URL...",
                    flush=True,
                )

                # The card is currently in the DOM, so click its own
                # Share button. Do not reuse a Share URL from another card.
                url = get_post_url(driver, scroll_container, signature)
                post['post_url'] = url
                processed.add(signature)

                if url:
                    print(f"    URL: {url}")
                else:
                    print("    URL: NOT FOUND")

                # After closing the Share popup, the card can become stale.
                time.sleep(0.4)

            except (StaleElementReferenceException, WebDriverException):
                continue
            except Exception as exc:
                try:
                    post['post_url'] = ''
                except Exception:
                    pass
                print(f"    ERROR: {exc}")

        # Stop once every discovered post has been processed.
        if len(processed) >= len(target):
            print("All discovered post URLs processed.")
            break

        try:
            state = driver.execute_script(r"""
                const e = arguments[0];
                return {
                    top: e.scrollTop,
                    height: e.scrollHeight,
                    client: e.clientHeight
                };
            """, scroll_container)
        except Exception:
            break

        top = state['top']
        height = state['height']
        client = state['client']
        at_bottom = top + client >= height - 10

        print(
            f"URL pass scroll {scroll_no:02d} | "
            f"top={int(top)} | height={int(height)} | "
            f"URLs={len(processed)}/{len(target)}",
            flush=True,
        )

        if at_bottom:
            stable_bottom += 1
        else:
            stable_bottom = 0

        if stable_bottom >= 3:
            break

        if previous_state == (top, height) and at_bottom:
            stable_bottom += 1
            if stable_bottom >= 3:
                break

        previous_state = (top, height)

        try:
            driver.execute_script(r"""
                const e = arguments[0];
                e.scrollTop = Math.min(
                    e.scrollTop + Math.floor(e.clientHeight * 0.70),
                    e.scrollHeight
                );
            """, scroll_container)
        except Exception:
            break

        time.sleep(1.1)

    missing = len(target) - len(processed)
    print(
        f"Finished URL pass: {len(processed)}/{len(target)} posts received a Share URL."
    )
    if missing:
        print(f"Share URLs still missing: {missing}")

    return posts


def make_dataframe(posts):
    import pandas as pd

    rows=[]
    for p in posts:
        image=p.get('imageUrl','').strip()
        video=p.get('videoUrl','').strip()
        rows.append({
            'date_posted': clean_text(p.get('date','')),
            'content': clean_text(p.get('content','')),
            'image_link': image,
            'video_link': video,
            'media_link': image or video,
            'post_link': p.get('post_url','').strip(),
        })
    df=pd.DataFrame(rows)
    if df.empty:
        return df
    df['_key']=df.apply(lambda r: norm(r['post_link']) if r['post_link'] else norm(r['date_posted'])+'||'+norm(r['content'][:1000]), axis=1)
    df=df.drop_duplicates('_key', keep='first').drop(columns='_key').reset_index(drop=True)
    return df


def save_posts_to_csv(posts, output_file):
    """Optional convenience helper - write posts to a CSV via make_dataframe()."""
    df = make_dataframe(posts)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    return df


def scrape_business_updates(
    business_query,
    headless=None,
    max_scrolls=None,
    scroll_pause=None,
    max_posts=None,
    fetch_post_urls=True,
    output_csv=None,
):
    """
    Main module entry point.

    Opens a Google search for `business_query`, opens its Knowledge Panel
    "Updates" popup, scrolls to load every available post, and (by default)
    visits each post's Share dialog to grab its post_url.

    Returns a list of post dicts, each shaped like:
        {
            "date": "...", "businessName": "...", "content": "...",
            "imageUrl": "...", "videoUrl": "...", "signature": "...",
            "post_url": "...",
        }

    Pass output_csv="some.csv" to also write results to disk via
    save_posts_to_csv(). Set fetch_post_urls=False to skip the (slower)
    per-post Share-URL pass and just return the loaded cards.
    """
    section("GOOGLE BUSINESS UPDATES SCRAPER")
    driver = create_driver(headless=headless)
    try:
        google_search(driver, business_query)
        popup = open_updates(driver)
        scroll_container = find_scroll_container(driver, popup)
        posts = load_all_posts(
            driver, scroll_container,
            max_scrolls=max_scrolls, scroll_pause=scroll_pause, max_posts=max_posts,
        )
        if not posts:
            print("No update cards found.")
            return []

        section("CARD PREVIEW")
        for i, p in enumerate(posts, 1):
            print(f"{i:02d}. {p['date']} | {p['content'][:180].replace(chr(10), ' ')}")

        if fetch_post_urls:
            posts = scrape_post_urls(driver, scroll_container, posts, max_scrolls=max_scrolls)

        if output_csv:
            df = save_posts_to_csv(posts, output_csv)
            section("FINAL RESULTS")
            print(f"Total unique posts: {len(df)}")
            print(f"Saved to: {output_csv}")

        return posts
    finally:
        try:
            driver.quit()
        except Exception:
            pass    