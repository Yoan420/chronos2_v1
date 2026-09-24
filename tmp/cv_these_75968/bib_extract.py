from pathlib import Path
import re
import unicodedata
from pypdf import PdfReader

BASE = Path(r"C:\Users\BQ6757\chronos2_v1\tmp\cv_these_75968")
SOURCE = Path(r"C:\Users\BQ6757\Downloads\Memoire_Yoan_Kesraoui.pdf")
reader = PdfReader(SOURCE)
assert len(reader.pages) == 33

first_authors = [
    ("Ansari", "Abdul Fatir"), ("Ansari", "Abdul Fatir"), ("Auer", "Andreas"),
    ("Ba", "Jimmy Lei"), ("Bommasani", "Rishi"), ("Das", "Abhimanyu"),
    ("Garza", "Azul"), ("Gneiting", "Tilmann"), ("Google Research", ""),
    ("Guibert", "Loïc"), ("Hu", "Edward J."), ("Hyndman", "Rob J."),
    ("Hyndman", "Rob J."), ("Jain", "Ayush"), ("Karaouli", "Nouha"),
    ("Khwaja", "Emaad"), ("Kim", "Taesung"), ("Laglil", "Morad"),
    ("Li", "Hongkai"), ("Lim", "Bryan"), ("Lipman", "Yaron"),
    ("Liu", "Yong"), ("Liu", "Chenghao"), ("Liu", "Xu"),
    ("Liu", "Yong"), ("Liu", "Yong"), ("Meyer", "Marcel"),
    ("Nie", "Yuqi"), ("Perez-Diaz", "Alvaro"), ("Qiao", "Zhongzheng"),
    ("Salesforce AI Research", ""), ("Salinas", "David"), ("Shchur", "Oleksandr"),
    ("Shi", "Xiaoming"), ("Vaswani", "Ashish"), ("Wan", "Shu"),
    ("Woo", "Gerald"), ("Wu", "Haixu"), ("Zeng", "Ailing"),
    ("Zhang", "Jiawen"), ("Zhou", "Haoyi"),
]

end_marker = re.compile(r"DOI\s*:\s*\S+|Source en ligne|arXiv\s+:\s*2108\.07258")
entries = []
original_links = []

for page_index in (30, 31, 32):
    page = reader.pages[page_index]
    raw = page.extract_text()
    if page_index == 30:
        raw = raw[raw.index("Abdul Fatir Ansari"):]
    else:
        raw = re.sub(r"^\s*\d+\s*\n", "", raw)
    links = []
    for annotation in page.get("/Annots", []):
        obj = annotation.get_object()
        action = obj.get("/A", {})
        uri = action.get("/URI")
        if uri:
            links.append((float(obj["/Rect"][3]), str(uri)))
    links.sort(reverse=True)
    parts = []
    start = 0
    for match in end_marker.finditer(raw):
        part = raw[start:match.end()]
        part = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", part)
        part = re.sub(r"(?<=\d)–\s*\n\s*(?=\d)", "–", part)
        part = re.sub(r"\s+", " ", part).strip()
        part = part.replace(" — ", " — ").replace("ci -dessous", "ci-dessous")
        parts.append(part)
        start = match.end()
    assert not raw[start:].strip(), (page_index + 1, raw[start:])
    assert len(parts) == len(links), (page_index + 1, len(parts), len(links))
    for part, (_, uri) in zip(parts, links):
        author = first_authors[len(entries)]
        year = int(re.search(r"\((\d{4})\)", part).group(1))
        if "Source en ligne" in part:
            part = part.replace("Source en ligne", f"[Consulter la source]({uri})")
        elif "DOI" in part:
            part = re.sub(r"DOI\s*:\s*(\S+)", lambda m: f"DOI : [{m.group(1)}]({uri})", part)
        else:
            part = part.replace("arXiv : 2108.07258", f"arXiv : [2108.07258]({uri})")
        if not part.endswith("."):
            part += "."
        entries.append((author[0], author[1], year, part))
        original_links.append(uri)

assert len(entries) == len(first_authors) == 41

additions = [
    ("Hollmann", "Noah", 2025,
     "Noah Hollmann ; Samuel Müller ; Lennart Purucker ; Arjun Krishnakumar ; Max Körfer ; Shi Bin Hoo ; Robin Tibor Schirrmeister ; Frank Hutter (2025). Accurate predictions on small data with a tabular foundation model. Nature, 637, 319–326. DOI : [10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6)."),
    ("Qu", "Jingang", 2025,
     "Jingang Qu ; David Holzmüller ; Gaël Varoquaux ; Marine Le Morvan (2025). TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. Proceedings of the 42nd International Conference on Machine Learning, PMLR, 267, 50817–50847. [Consulter la source](https://proceedings.mlr.press/v267/qu25d.html)."),
    ("Qu", "Jingang", 2026,
     "Jingang Qu ; David Holzmüller ; Gaël Varoquaux ; Marine Le Morvan (2026). TabICLv2: A better, faster, scalable, and open tabular foundation model. Prépublication, arXiv:2602.11139v1, 11 février 2026. [Consulter la version citée](https://arxiv.org/abs/2602.11139v1)."),
]
entries.extend(additions)

def fold(value):
    return "".join(ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch)).casefold()

entries.sort(key=lambda item: (fold(item[0]), fold(item[1]), item[2], fold(item[3])))
intro = (
    "# Bibliographie\n\n"
    "Les références sont classées selon le nom du premier auteur, puis son prénom et l’année. "
    "Les 41 références initiales et leurs liens proviennent des pages 31 à 33 du mémoire fourni ; "
    "les URL ont été récupérées dans les annotations du PDF. Les mentions de consultation des 9 et 10 septembre 2026 "
    "et les statuts de publication sont ceux du document d’origine et n’ont pas fait l’objet d’une nouvelle vérification exhaustive. "
    "Les trois références ajoutées sur TabPFN, TabICL et TabICLv2 ont été vérifiées dans leurs sources primaires le 16 septembre 2026. "
    "Les auteurs et titres d’origine sont conservés ; les coupures de lignes ont été normalisées.\n\n"
)
result = intro + "\n\n".join(entry[3] for entry in entries) + "\n"
assert "Source en ligne" not in result
assert len(re.findall(r"\]\(https?://", result)) == 44
assert all(uri in result for uri in original_links)
BASE.joinpath("bibliographie.md").write_text(result, encoding="utf-8")

for number, page_index in enumerate((6, 7, 13), 1):
    images = list(reader.pages[page_index].images)
    assert len(images) == 1
    image = images[0]
    assert image.name.lower().endswith(".png")
    target = BASE / f"figure_{number}.png"
    target.write_bytes(image.data)
    print(f"{target.name}: page {page_index + 1}, {image.image.size}, {len(image.data)} bytes")

print(f"Bibliography: {len(entries)} entries, {len(original_links)} original links, 3 added links, 44 total links")
print("Order:", ", ".join(f"{entry[0]} {entry[2]}" for entry in entries))
