"""Scrape Chula Engineering faculty profiles and write only professors.json.

Run from the repository root:
    python3 scripts/scrape_professors_only.py

Optionally scrape one department:
    python3 scripts/scrape_professors_only.py --department mechanical_engineering

The script uses the official department directory URLs below and reads the
existing professor records,
follows individual profile hrefs on official Chula Engineering hosts, and
updates only data/processed/professors.json. Existing non-empty email and
research-interest values are preserved.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup, Tag


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROFESSORS_PATH = DATA / "professors.json"
DEPARTMENT_DIRECTORIES = {
    "computer_engineering": "https://www.cp.eng.chula.ac.th/about/faculty",
    "nuclear_engineering": "https://ne.eng.chula.ac.th/wp/index.php/people/",
    "chemical_engineering": "https://chem.eng.chula.ac.th/people-faculty/",
    "mechanical_engineering": "https://me.eng.chula.ac.th/mecu-faculty-members/",
    "electrical_engineering": "https://ee.eng.chula.ac.th/faculty/",
    "civil_engineering": "https://civil.eng.chula.ac.th/web/ce-department-faculty-directory/",
    "metallurgical_engineering": "https://metal.eng.chula.ac.th/list/lecturer",
    "survey_engineering": "https://sv.eng.chula.ac.th/เกี่ยวกับวิศวกรรมสำรวจ/คณาจารย์และเจ้าหน้าที่/",
    "environmental_sustainable_engineering": "https://env.eng.chula.ac.th/?page_id=9537",
    "mining_petroleum_engineering": "https://mining.eng.chula.ac.th/staff/",
    "water_resources_engineering": "https://water.eng.chula.ac.th/faculty/",
    "industrial_engineering": "https://ienext.eng.chula.ac.th/?page_id=3429",
}
USER_AGENT = "ChulaEngineeringFacultyDirectory/1.0 (academic directory indexing)"
REQUEST_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 25.0

IGNORED_LABELS = {
    "read more", "view profile", "profile", "more", "more info", "more information",
    "see more", "click here", "learn more", "details",
    "faculty", "faculty members", "our people", "people", "home", "email",
    "website", "scopus", "google scholar", "linkedin", "facebook", "twitter",
    "next", "previous", "ดูเพิ่มเติม", "อ่านเพิ่มเติม", "เพิ่มเติม", "ประวัติ",
    "คณาจารย์", "บุคลากร", "อาจารย์", "หน้าแรก",
}
NON_PROFILE_PATH_PARTS = {
    "category", "tag", "author", "feed", "search", "events", "news", "contact",
    "admission", "program", "programs", "research", "research-center", "lab",
    "laboratory", "privacy-policy", "facebook.com", "linkedin.com", "youtube.com",
}
NON_PERSON_WORDS = {
    "department", "engineering", "faculty", "staff", "program", "programme",
    "research", "admission", "contact", "people", "academic", "directory",
}
INTEREST_LABELS = (
    "research interests", "research interest", "research areas", "research area",
    "research expertise", "research focus", "major area of scientific interest",
    "areas of expertise", "ความสนใจงานวิจัย", "งานวิจัยที่สนใจ", "ความเชี่ยวชาญ",
)


def is_interest_label(label: str) -> bool:
    label = clean_text(label).casefold().rstrip(":")
    return label == "research" or any(term in label for term in INTEREST_LABELS)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def normalized_name(value: str | None) -> str:
    """Normalize names to match directory names and profile headings."""
    value = clean_text(value).casefold()
    value = re.sub(r"\b(professor|prof|associate|assoc|assistant|asst|dr|ph\.?d|d\.?eng|d\.?sc)\b", " ", value)
    value = re.sub(r"[^a-z0-9\u0e00-\u0e7f]+", " ", value)
    return clean_text(value)


def is_person_name(value: str | None) -> bool:
    value = clean_text(value)
    if not value or value.casefold().strip(" .:") in IGNORED_LABELS:
        return False
    words = re.findall(r"[A-Za-z\u0e00-\u0e7f][A-Za-z\u0e00-\u0e7f'.-]*", value)
    if len(words) < 2 or len(value) > 100:
        return False
    if any(word.casefold().strip(".-") in NON_PERSON_WORDS for word in words):
        return False
    return bool(re.search(r"[A-Za-z\u0e00-\u0e7f]", value))


def official_engineering_host(url: str) -> bool:
    host = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    return host == "chula.ac.th" or host.endswith(".chula.ac.th")


def path_is_profile_candidate(url: str, directory_url: str) -> bool:
    parsed = urlsplit(url)
    directory = urlsplit(directory_url)
    if parsed.scheme not in {"http", "https"} or not official_engineering_host(url):
        return False
    candidate_base = (parsed.scheme.casefold(), parsed.netloc.casefold().removeprefix("www."), parsed.path.rstrip("/"))
    directory_base = (directory.scheme.casefold(), directory.netloc.casefold().removeprefix("www."), directory.path.rstrip("/"))
    if candidate_base == directory_base and (not parsed.query or parsed.query == directory.query):
        return False
    path = parsed.path.casefold().rstrip("/")
    directory_path = directory.path.casefold().rstrip("/")
    if (path == directory_path and (not parsed.query or parsed.query == directory.query)) or not path:
        return False
    # Discard navigation and section pages; keep paths such as /profile/39.
    parts = {part for part in path.split("/") if part}
    if parts & NON_PROFILE_PATH_PARTS:
        return False
    if parsed.path.lower().endswith((".pdf", ".jpg", ".jpeg", ".png", ".doc", ".docx", ".zip")):
        return False
    return True


def likely_name_from_block(anchor: Tag, existing_names: list[str]) -> str | None:
    anchor_text = clean_text(anchor.get_text(" ", strip=True))
    if is_person_name(anchor_text):
        return anchor_text

    # Profile cards commonly use an image or a “Read More” link; inspect the
    # nearest card headings and known roster names in that same block.
    block: Tag | None = anchor
    for _ in range(6):
        if block is None or not isinstance(block, Tag):
            break
        block_text = clean_text(block.get_text(" ", strip=True))
        heading = block.find(["h1", "h2", "h3", "h4", "h5"])
        if heading:
            heading_text = clean_text(heading.get_text(" ", strip=True))
            if is_person_name(heading_text):
                return heading_text
        normalized_block = normalized_name(block_text)
        for known_name in existing_names:
            normalized_known = normalized_name(known_name)
            if normalized_known and normalized_known in normalized_block:
                return known_name
        block = block.parent if isinstance(block.parent, Tag) else None

    for attr in ("aria-label", "title", "alt"):
        text = clean_text(anchor.get(attr))
        if is_person_name(text):
            return text
    return None


def profile_links(
    html: str,
    page_url: str,
    existing_names: list[str],
) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, dict[str, str]] = {}
    for anchor in soup.find_all("a", href=True):
        href = clean_text(anchor.get("href"))
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute_url = urldefrag(urljoin(page_url, href))[0]
        if not path_is_profile_candidate(absolute_url, page_url):
            continue
        name = likely_name_from_block(anchor, existing_names)
        if not name:
            continue
        found.setdefault(absolute_url, {"name_en": name, "profile_url": absolute_url})
    return list(found.values())


def split_interest_text(value: str) -> list[str]:
    value = clean_text(value)
    if not value:
        return []
    # Keep sentences intact; split only on common list separators.
    chunks = re.split(r"\s*(?:\u2022|\u25cf|\u25aa|\||;|\n)\s*", value)
    return list(dict.fromkeys(clean_text(item).strip(" ,:-") for item in chunks if clean_text(item).strip(" ,:-")))


def extract_labeled_interests(soup: BeautifulSoup, page_text: str) -> tuple[list[str], str | None]:
    results: list[str] = []
    descriptions: list[str] = []
    for label_node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "strong", "b", "dt", "th"]):
        label = clean_text(label_node.get_text(" ", strip=True)).casefold().rstrip(":")
        if not label or not is_interest_label(label):
            continue
        content = label_node.find_next(["p", "ul", "ol", "dd", "td"])
        if not content:
            continue
        value = clean_text(content.get_text("; " if content.name in {"ul", "ol"} else " ", strip=True))
        results.extend(split_interest_text(value))
        if len(value) > 180:
            descriptions.append(value)

    # Some WordPress profiles print the heading and value on separate lines.
    lines = [clean_text(line) for line in page_text.splitlines() if clean_text(line)]
    for index, line in enumerate(lines[:-1]):
        low = line.casefold().rstrip(":")
        if is_interest_label(low):
            results.extend(split_interest_text(lines[index + 1]))

    results = list(dict.fromkeys(item for item in results if item))
    return results, "\n".join(dict.fromkeys(descriptions)) or None


def parse_profile(html: str, url: str, fallback_name: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    page_text = soup.get_text("\n", strip=True)
    heading = soup.find("h1") or soup.find("h2") or soup.title
    name = clean_text(heading.get_text(" ", strip=True)) if heading else fallback_name
    if not is_person_name(name) or name.casefold() in {"faculty", "faculty members", "profile", "people"}:
        name = fallback_name

    email = None
    mailto = soup.select_one('a[href^="mailto:"]')
    if mailto:
        email = clean_text(mailto.get("href", "").removeprefix("mailto:").split("?", 1)[0])
    else:
        labelled = re.search(r"(?:e-?mail|อีเมล)\s*:?\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", page_text, re.I)
        if labelled:
            email = labelled.group(1)

    phone = None
    tel = soup.select_one('a[href^="tel:"]')
    if tel:
        phone = clean_text(tel.get("href", "").removeprefix("tel:"))

    interests, description = extract_labeled_interests(soup, page_text)
    return {
        "name_en": name,
        "email": email,
        "phone": phone,
        "research_interests": interests,
        "research_description": description,
        "profile_url": url,
    }


def scrape_department(
    client: httpx.Client,
    department: dict[str, Any],
    existing_records: list[dict[str, Any]],
    last_request: list[float],
    delay_seconds: float,
) -> list[dict[str, Any]]:
    directory_url = department.get("faculty_directory_url")
    if not directory_url:
        print(f"[skip] {department['department_id']}: no faculty_directory_url", file=sys.stderr)
        return []
    known_names = [
        p.get("name_en", "") for p in existing_records
        if p.get("department_id") == department["department_id"] and p.get("name_en")
    ]

    def get(url: str) -> tuple[str, str]:
        wait = delay_seconds - (time.monotonic() - last_request[0])
        if wait > 0:
            time.sleep(wait)
        response = client.get(url)
        response.raise_for_status()
        last_request[0] = time.monotonic()
        return response.text, str(response.url)

    try:
        listing_html, final_listing_url = get(directory_url)
    except (httpx.HTTPError, OSError) as exc:
        print(f"[error] {department['department_id']} directory {directory_url}: {exc}", file=sys.stderr)
        return []

    candidates = profile_links(listing_html, final_listing_url, known_names)
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            html, final_profile_url = get(candidate["profile_url"])
        except (httpx.HTTPError, OSError) as exc:
            print(f"[error] {department['department_id']} profile {candidate['profile_url']}: {exc}", file=sys.stderr)
            continue
        profile = parse_profile(html, final_profile_url, candidate["name_en"])
        profile.update({
            "department_id": department["department_id"],
            "source_url": final_listing_url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "verification_status": "possibly_current",
            "original_values": {
                "directory_name": candidate["name_en"],
                "profile_source_url": final_profile_url,
                "research_interest_source_url": final_profile_url if profile["research_interests"] else None,
                "profile_checked": True,
                "research_info_status": "explicit_profile_section" if profile["research_interests"] else "profile_checked_no_explicit_interest",
            },
        })
        records.append(profile)
    print(f"[done] {department['department_id']}: {len(records)} profile pages read", file=sys.stderr)
    return records


def merge_records(existing: list[dict[str, Any]], scraped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    by_name: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_profile: dict[tuple[str, str], dict[str, Any]] = {}
    for record in merged:
        department_id = record.get("department_id", "")
        name_key = (department_id, normalized_name(record.get("name_en")))
        if name_key[1]:
            by_name.setdefault(name_key, []).append(record)
        profile_key = (department_id, urldefrag(record.get("profile_url", ""))[0].rstrip("/"))
        if profile_key[1]:
            by_profile[profile_key] = record

    for fresh in scraped:
        department_id = fresh["department_id"]
        profile_key = (department_id, urldefrag(fresh.get("profile_url", ""))[0].rstrip("/"))
        name_key = (department_id, normalized_name(fresh.get("name_en")))
        directory_name = fresh.get("original_values", {}).get("directory_name")
        directory_name_key = (department_id, normalized_name(directory_name))
        match = by_profile.get(profile_key)
        if match is None:
            matches = by_name.get(name_key, []) or by_name.get(directory_name_key, [])
            match = matches[0] if matches else None

        if match is None:
            name_slug = re.sub(r"[^a-z0-9]+", "_", normalized_name(fresh.get("name_en"))).strip("_")[:64]
            record = {
                "professor_id": f"{department_id}_{name_slug or 'profile'}",
                "name_th": None,
                "name_en": fresh.get("name_en"),
                "academic_title": None,
                "department_id": department_id,
                "position": None,
                "email": fresh.get("email"),
                "phone": fresh.get("phone"),
                "research_interests": fresh.get("research_interests", []),
                "research_description": fresh.get("research_description"),
                "profile_url": fresh.get("profile_url"),
                "source_url": fresh.get("source_url"),
                "scraped_at": fresh.get("scraped_at"),
                "verification_status": fresh.get("verification_status", "possibly_current"),
                "original_values": fresh.get("original_values", {}),
            }
            merged.append(record)
            by_name.setdefault(name_key, []).append(record)
            by_profile[profile_key] = record
            continue

        # Keep curated values already present; use the individual profile to
        # fill blanks and replace a directory URL with the direct href.
        match["profile_url"] = fresh.get("profile_url") or match.get("profile_url")
        if not match.get("email") and fresh.get("email"):
            match["email"] = fresh["email"]
            match.setdefault("original_values", {})["email_source"] = "official_individual_profile"
        if not match.get("phone") and fresh.get("phone"):
            match["phone"] = fresh["phone"]
        if not match.get("research_interests") and fresh.get("research_interests"):
            match["research_interests"] = fresh["research_interests"]
            match.setdefault("original_values", {})["research_interest_source_url"] = fresh["profile_url"]
            match["original_values"]["research_info_status"] = "explicit_profile_section"
        if not match.get("research_description") and fresh.get("research_description"):
            match["research_description"] = fresh["research_description"]
        original = match.setdefault("original_values", {})
        original["profile_source_url"] = fresh.get("profile_url")
        original["profile_checked"] = True
        original.setdefault("directory_name", fresh.get("original_values", {}).get("directory_name"))
        match["scraped_at"] = fresh.get("scraped_at")
        by_profile[profile_key] = match
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--department", help="Scrape only this department_id")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS, help="Seconds between HTTP requests (default: 1)")
    args = parser.parse_args()

    existing = read_json(PROFESSORS_PATH, [])
    if not isinstance(existing, list):
        raise SystemExit(f"Expected a JSON array in {PROFESSORS_PATH}")
    departments = [
        {"department_id": department_id, "faculty_directory_url": url}
        for department_id, url in DEPARTMENT_DIRECTORIES.items()
    ]
    selected = departments
    if args.department:
        selected = [d for d in departments if d.get("department_id") == args.department]
        if not selected:
            raise SystemExit(f"Unknown department_id: {args.department}")

    timeout = httpx.Timeout(TIMEOUT_SECONDS)
    last_request = [0.0]
    scraped: list[dict[str, Any]] = []
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for department in selected:
            scraped.extend(scrape_department(client, department, existing, last_request, max(0.0, args.delay)))

    if scraped:
        result = merge_records(existing, scraped)
        PROFESSORS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {PROFESSORS_PATH} with {len(scraped)} profile pages; total records: {len(result)}")
    else:
        print("No profile pages were scraped; professors.json was left unchanged.", file=sys.stderr)


if __name__ == "__main__":
    main()
