"""
Collect faculty + publication data for Lovely Professional University (LPU).

Method A (recommended): OpenAlex API - free, legal, structured, no HTML scraping.
Method B (optional)   : scrape LPU's own faculty pages (selectors must be filled in by you).

Install:  pip install requests pandas beautifulsoup4
Run:      python collect_lpu_data.py
Output:   lpu_faculty.csv  (one row per researcher: name, topics, paper titles, counts)
          lpu_works.csv    (one row per paper)

NOTE: written without being able to reach the internet from my sandbox, so test it on
your machine and adjust if the API response fields differ slightly.
"""
import time
import requests
import pandas as pd
from collections import defaultdict

API = "https://api.openalex.org"
import os
MAILTO = os.environ.get("OPENALEX_EMAIL", "your_email@example.com")
API_KEY = os.environ.get("OPENALEX_API_KEY", "")

def get(url, params=None):
    params = dict(params or {}, mailto=MAILTO, api_key=API_KEY)
    r = requests.get(url, params=params, timeout=30)
    if r.status_code in (401, 403):
        raise SystemExit("OpenAlex rejected the request. Check that API_KEY is set correctly "
                         "(get a free key at https://openalex.org/settings/api).")
    r.raise_for_status()
    return r.json()


def find_institution_id(name="Lovely Professional University"):
    res = get(f"{API}/institutions", {"search": name})["results"]
    for r in res[:5]:
        print(" candidate:", r["id"], "|", r["display_name"], "|", r.get("country_code"))
    return res[0]["id"]          # check the printed candidates and pick the right one if needed


def abstract_from_index(inv):
    """OpenAlex stores abstracts as an inverted index; rebuild plain text."""
    if not inv:
        return ""
    words = {}
    for w, positions in inv.items():
        for p in positions:
            words[p] = w
    return " ".join(words[i] for i in sorted(words))


def fetch_works(inst_id, since_year=2018, max_pages=50):
    """All papers with at least one LPU-affiliated author (cursor pagination)."""
    cursor, works = "*", []
    for _ in range(max_pages):
        data = get(f"{API}/works", {
            "filter": f"authorships.institutions.id:{inst_id},from_publication_date:{since_year}-01-01",
            "per-page": 200, "cursor": cursor,
        })
        works += data["results"]
        cursor = data["meta"].get("next_cursor")
        print(f"  fetched {len(works)} works")
        if not cursor:
            break
        time.sleep(0.2)          # be polite
    return works


def build_tables(works, inst_id):
    paper_rows = []
    people = defaultdict(lambda: {"name": "", "titles": [], "abstracts": [], "topics": [], "n_papers": 0})
    short_id = inst_id.split("/")[-1]

    for w in works:
        topics = [t["display_name"] for t in (w.get("topics") or [])]
        abstract = abstract_from_index(w.get("abstract_inverted_index"))
        paper_rows.append({"id": w["id"], "title": w["title"], "year": w["publication_year"],
                           "topics": "; ".join(topics), "abstract": abstract,
                           "cited_by": w.get("cited_by_count")})
        for a in w.get("authorships", []):
            if any(i["id"].endswith(short_id) for i in a.get("institutions", [])):
                p = people[a["author"]["id"]]
                p["name"] = a["author"]["display_name"]
                p["titles"].append(w["title"] or "")
                p["abstracts"].append(abstract)
                p["topics"] += topics
                p["n_papers"] += 1

    fac = pd.DataFrame([{
        "faculty_id": k, "name": v["name"], "n_papers": v["n_papers"],
        "publications": " | ".join(v["titles"]),
        "research": " ".join(v["abstracts"])[:5000],
        "top_topics": "; ".join(pd.Series(v["topics"]).value_counts().head(8).index),
    } for k, v in people.items()])
    return fac.sort_values("n_papers", ascending=False), pd.DataFrame(paper_rows)


# ----------------------------------------------------------------------------
# Method B: scrape the university's own faculty/profile pages (OPTIONAL)
# Steps: 1) open the page in Chrome  2) right-click > Inspect to find the HTML tags
#        3) check <site>/robots.txt and the site's terms  4) fill in the selectors below.
# ----------------------------------------------------------------------------
def scrape_faculty_page(url):
    from bs4 import BeautifulSoup
    html = requests.get(url, headers={"User-Agent": "student-research-project"}, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for card in soup.select("CSS_SELECTOR_FOR_ONE_FACULTY_CARD"):          # <- you fill this in
        rows.append({
            "name": card.select_one("CSS_FOR_NAME").get_text(strip=True),
            "department": card.select_one("CSS_FOR_DEPARTMENT").get_text(strip=True),
            "research_interests": card.select_one("CSS_FOR_INTERESTS").get_text(strip=True),
        })
        time.sleep(1)                                                       # don't hammer the server
    return pd.DataFrame(rows)


if __name__ == "__main__":
    inst = find_institution_id()
    print("Using institution:", inst)
    works = fetch_works(inst)
    faculty, papers = build_tables(works, inst)
    faculty.to_csv("lpu_faculty.csv", index=False)
    papers.to_csv("lpu_works.csv", index=False)
    print(f"Saved {len(faculty)} researchers and {len(papers)} papers.")
