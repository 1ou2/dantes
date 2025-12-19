import os
from pathlib import Path
import sys
import re
import textwrap
from typing import List, Tuple, Optional

def preprocess(indir="data/raw/gutenberg", outdir="data/preprocessed/gutenberg"):
    """Preprocess Project Gutenberg text files by removing headers and footers.
    Args:
        indir (str): Input directory containing raw Gutenberg text files.
        outdir (str): Output directory to save preprocessed text files.
    """
    # Define files to check
    files_to_check = list(Path(indir).glob("*.txt"))
    preprocessed_dir = Path(outdir)
    # create preprocessed dir if it doesn't exist
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    for file_path in files_to_check:
        print(f"Preprocessing {file_path}...")
        startline = 0
        endline = -1
        # opening file with utf-8-sig encoding to avoid BOM issues
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            # only keep lines that are not empty
            lines = [line for line in lines if line.strip() != ""]

            # looking for line : *** START OF THE PROJECT GUTENBERG EBOOK
            for i, line in enumerate(lines):
                if line.startswith("***"):
                    startline = i +1
                    break

            if startline != 0:
                # after what should be the start line we still have other comments in the subsequent lines
                headers = ["produced", "distributed", "proofreading","etext","file","by","http","is","mobipocket"
                "online","available","e-text","the", "Bibliothèque",
                "from","(http","of","at","you","before","whatsoever", "Text", "and the", "we",
                "this", "is", "made","encoded", "note:"]
                for i, line in enumerate(lines[startline:]):
                    if line.strip() == "":
                        startline += 1
                    else:
                        start_with_header = False
                        # check if line starts with any of the headers
                        for header in headers:
                            if line.lower().startswith(header):
                                startline += 1
                                start_with_header = True
                        # did not find a line starting with a header, nor an empty line
                        # we should be at the start of the book
                        if not start_with_header:
                            break

                # looking for line : *** END OF THE PROJECT GUTENBERG EBOOK
                for i, line in enumerate(lines[startline:]):
                    if line.startswith("***"):
                        endline = i + startline
                        break


            # write all lines after startline to file
            # get basename of file and write to "preprocessed" dir
            basename = file_path.name
            preprocessed_path = Path(preprocessed_dir) / basename
            # write preprocessed file using utf-8 encoding
            with open(preprocessed_path, 'w', encoding='utf-8') as f:
                f.writelines(lines[startline:endline])

def download_gutenberg_book(book_id,data_dir):
    """Download a book from Project Gutenberg by its ID.
     Args:
        book_id (int): The Project Gutenberg book ID.
        data_dir (str): The directory to save the downloaded book.
    """
    gutenberg_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    filename = f"pg{book_id}.txt"
    # check if file already exists
    if os.path.exists(f"{data_dir}/{filename}"):
        print(f"File {filename} already exists")
        return
    os.system(f"wget {gutenberg_url} -P {data_dir}")




"""
Reformatteur de texte Gutenberg (corrigé pour dialogues) :
- Supprime les retours à la ligne durs (~80 colonnes) au sein des paragraphes.
- Conserve les fins de paragraphes via heuristique (dernière ligne courte punctuée).
- Dialogues :
    * Toute ligne qui commence par un marqueur de dialogue (—, --, –, -, «) démarre une réplique.
    * Les lignes suivantes qui ne commencent PAS par un marqueur sont rattachées à la même réplique
      si ce n'est pas clairement un nouveau départ de phrase.
      Heuristiques de rattachement :
        - guillemets « non refermés dans la réplique => poursuivre,
        - OU la ligne précédente est probablement "cassée" (longueur ~ wrap source),
        - OU la ligne courante commence par une minuscule,
        - OU la ligne précédente ne se termine PAS par une ponctuation forte (., !, ?, …, »).
    * La réplique est "flushée" quand on voit un nouveau marqueur, une ligne vide,
      ou si l'heuristique conclut que la ligne suivante est un nouveau départ.

- Gère les césures (tiret en fin de ligne + minuscule ensuite).
- Option pour enlever en-têtes/pieds Project Gutenberg.

Usage :
    python reflow_gutenberg.py -i input.txt -w 100 > output.txt
"""
DIALOGUE_MARKERS_DEFAULT = ("—", "--", "–", "-", "«")

START_PATTERNS = (
    re.compile(r"^\*{3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK", re.I),
)
END_PATTERNS = (
    re.compile(r"^\*{3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK", re.I),
)

SENTENCE_END_RE = re.compile(r"[.!?…»][”’'\)\]]*$")

def looks_like_dialogue(line: str, markers=DIALOGUE_MARKERS_DEFAULT) -> bool:
    s = line.lstrip()
    return any(s.startswith(m) for m in markers)

def strip_gutenberg_headers(text: str) -> str:
    lines = text.splitlines()
    started = False
    out = []
    for ln in lines:
        if not started:
            if any(p.search(ln) for p in START_PATTERNS):
                started = True
            continue
        else:
            if any(p.search(ln) for p in END_PATTERNS):
                break
            out.append(ln)
    if not out:
        return text
    return "\n".join(out)

def should_end_paragraph(raw_line: str, source_wrap: int) -> bool:
    s = raw_line.rstrip()
    if not s:
        return True  # ligne vide = séparation de paragraphe
    short_threshold = int(source_wrap * 0.75)
    return (len(s) <= short_threshold) and bool(SENTENCE_END_RE.search(s))

def join_with_hyphenation(prev: str, curr: str) -> str:
    if not prev:
        return curr.strip()
    prev_stripped = prev.rstrip()
    curr_stripped = curr.strip()
    if prev_stripped.endswith("-") and not prev_stripped.endswith("--"):
        if curr_stripped and curr_stripped[0].islower():
            return prev_stripped[:-1] + curr_stripped
    return prev_stripped + " " + curr_stripped

def reflow_text(
    text: str,
    wrap_width: int = 0,
    source_wrap: int = 80,
    dialogue_markers: Tuple[str, ...] = DIALOGUE_MARKERS_DEFAULT,
    strip_headers: bool = False,
) -> str:
    if strip_headers:
        text = strip_gutenberg_headers(text)

    lines = text.splitlines()
    paragraphs: List[Tuple[str, str]] = []  # (kind, content) kind in {"text", "dialogue", "blank"}

    # Buffers
    text_buf = ""  # accumulation de texte narratif
    dialog_buf: Optional[dict] = None  # {"content": str, "last_raw": str, "in_quote": bool}

    def flush_text():
        nonlocal text_buf
        if text_buf.strip():
            paragraphs.append(("text", text_buf.strip()))
        text_buf = ""

    def flush_dialog():
        nonlocal dialog_buf
        if dialog_buf and dialog_buf["content"].strip():
            paragraphs.append(("dialogue", dialog_buf["content"].strip()))
        dialog_buf = None

    def update_in_quote(s: str) -> bool:
        return s.count("«") > s.count("»")

    def likely_wrapped(prev_raw: str) -> bool:
        # Ligne précédente proche de la largeur source => probable retour dur
        return len(prev_raw.rstrip()) >= int(source_wrap * 0.90)

    for raw in lines:
        s = raw.strip()

        # Séparateur net
        if not s:
            flush_text()
            flush_dialog()
            paragraphs.append(("blank", ""))
            continue

        if looks_like_dialogue(s, dialogue_markers):
            # Nouvelle réplique
            flush_text()
            flush_dialog()
            dialog_buf = {
                "content": s,
                "last_raw": raw.rstrip(),
                "in_quote": update_in_quote(s),
            }
            continue

        # Ligne ne commençant PAS par un marqueur
        if dialog_buf is not None:
            # Décider si c'est la continuation de la réplique en cours
            last_raw = dialog_buf["last_raw"]
            prev_ends_sentence = bool(SENTENCE_END_RE.search(last_raw))
            first_char = s[0]
            begins_lower = first_char.islower()

            # Heuristique de continuation:
            #  - guillemet ouvert non refermé, OU
            #  - ligne précédente "cassée" à ~source_wrap, OU
            #  - la présente ligne commence par une minuscule, OU
            #  - la précédente ne se termine pas par ponctuation forte.
            if (
                dialog_buf["in_quote"]
                or likely_wrapped(last_raw)
                or begins_lower
                or not prev_ends_sentence
            ):
                dialog_buf["content"] = join_with_hyphenation(dialog_buf["content"], raw)
                dialog_buf["last_raw"] = raw.rstrip()
                dialog_buf["in_quote"] = update_in_quote(dialog_buf["content"])
            else:
                # On estime que c'est un nouveau départ (narration ou autre)
                flush_dialog()
                # Traiter comme texte narratif
                if not text_buf:
                    text_buf = s
                else:
                    text_buf = join_with_hyphenation(text_buf, raw)
                if should_end_paragraph(raw, source_wrap):
                    flush_text()
            continue

        # Texte narratif (hors dialogue)
        if not text_buf:
            text_buf = s
        else:
            text_buf = join_with_hyphenation(text_buf, raw)

        if should_end_paragraph(raw, source_wrap):
            flush_text()

    # Fin: vider les buffers
    flush_dialog()
    flush_text()

    # Compacte les blancs
    compacted: List[Tuple[str, str]] = []
    last_blank = False
    for kind, content in paragraphs:
        if kind == "blank":
            if not last_blank:
                compacted.append((kind, content))
            last_blank = True
        else:
            compacted.append((kind, content))
            last_blank = False

    # Sortie: rewrap du texte narratif uniquement
    out_lines: List[str] = []
    wrapper: Optional[textwrap.TextWrapper] = None
    if wrap_width and wrap_width > 0:
        wrapper = textwrap.TextWrapper(width=wrap_width, replace_whitespace=True, drop_whitespace=True)

    for kind, content in compacted:
        if kind == "blank":
            out_lines.append("")
        elif kind == "dialogue":
            # Laisser la réplique sur une ligne (sans rewrap)
            out_lines.append(content)
        else:  # "text"
            if wrapper:
                out_lines.append(wrapper.fill(content))
            else:
                out_lines.append(content)

    return "\n".join(out_lines).strip("\n")

if __name__ == "__main__":
    monte_cristo = [17989,17990,17991,17992]
    download_dir = "data/raw"
    preprocessed_dir = "data/preprocessed"
    output_dir = "data/gutenberg"

    for dir in [download_dir, preprocessed_dir, output_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for book_id in monte_cristo:
        download_gutenberg_book(book_id, download_dir)

    preprocess(indir=download_dir, outdir=preprocessed_dir)
    for book_id in monte_cristo:
        file = f"pg{book_id}.txt"
        preprocessed_path = Path(preprocessed_dir) / file
        out = Path(output_dir) / file

        with open(preprocessed_path, 'r', encoding='utf-8') as f:
            text = f.read()
            result = reflow_text(text,wrap_width=0,strip_headers=False)
            with open(out, 'w', encoding='utf-8') as outf:
                outf.write(result)