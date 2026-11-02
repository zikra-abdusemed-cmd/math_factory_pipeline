"""Read Open edX OLX exports and turn lesson pages into pipeline briefs.

MyMarian Grade 9 Physics stores the real teaching text in ``static/*.htm``
files and embeds them from HTML XBlocks via iframes. This loader follows
chapter → sequential → vertical → html, then strips visible text.
"""

from __future__ import annotations

import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET
from html import unescape


_STATIC_SRC = re.compile(r"""src\s*=\s*["']/static/([^"']+)["']""", re.I)
_SKIP_SEQUENTIAL = re.compile(r"quiz|interactive\s+tools|\btools\b", re.I)
_SKIP_VERTICAL = re.compile(r"\bquiz\b|assessment", re.I)
_BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "br", "blockquote", "table",
}


@dataclass(frozen=True)
class OlxBlock:
    url_name: str
    display_name: str
    kind: str
    text: str = ""
    children: tuple["OlxBlock", ...] = ()

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class OlxCourse:
    org: str
    course: str
    run: str
    display_name: str
    root: Path
    chapters: tuple[OlxBlock, ...] = ()

    @property
    def course_id(self) -> str:
        return f"course-v1:{self.org}+{self.course}+{self.run}"


@dataclass
class LessonBrief:
    title: str
    course_id: str
    unit_id: str
    unit_title: str
    section_ids: list[str] = field(default_factory=list)
    source_text: str = ""

    def extra_instructions(self, *, max_chars: int = 14000) -> str:
        body = self.source_text.strip()
        if len(body) > max_chars:
            body = body[: max_chars - 20].rsplit(" ", 1)[0] + "\n[truncated]"
        return (
            f"Teach this MyMarian LMS lesson. Use ONLY facts, examples, names, "
            f"and terms from the source text. Do not invent measurements, dates, "
            f"or engineering claims that are not written there.\n\n"
            f"Course: {self.title} ({self.course_id})\n"
            f"Unit: {self.unit_title}\n"
            f"Audience: Grade 9.\n"
            f"Dark theme; blue variables, yellow givens, green answers, orange highlights. "
            f"Schematic diagrams only. Camera movement only when it clarifies the idea. "
            f"visual_plan.parameters must contain numeric physics values only.\n\n"
            f"Source lesson text:\n{body}"
        )


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript"}:
            self._skip += 1
            return
        if self._skip:
            return
        if lowered in _BLOCK_TAGS:
            self._chunks.append("\n")
        if lowered == "img":
            alt = next((value for name, value in attrs if name == "alt" and value), "")
            if alt:
                self._chunks.append(alt.strip())

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
            return
        if not self._skip and lowered in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = unescape(data)
        if text.strip():
            self._chunks.append(text)

    def text(self) -> str:
        joined = " ".join(self._chunks)
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r"\n(?:[ \t]*\n)+", "\n", joined)
        return joined.strip()


def html_to_text(markup: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(markup)
    parser.close()
    return parser.text()


def _parse_xml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _attr(element: ET.Element, name: str, default: str = "") -> str:
    return (element.attrib.get(name) or default).strip()


def _child_url_names(element: ET.Element, tag: str) -> list[str]:
    names = []
    for child in list(element):
        local = child.tag.split("}")[-1]
        if local == tag:
            url_name = _attr(child, "url_name")
            if url_name:
                names.append(url_name)
    return names


def _locate_course_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.name.endswith((".tar.gz", ".tgz")):
        tmp = Path(tempfile.mkdtemp(prefix="olx_"))
        with tarfile.open(path, "r:gz") as archive:
            try:
                archive.extractall(tmp, filter="data")
            except TypeError:
                archive.extractall(tmp)
        path = tmp
    if path.is_file() and path.name == "course.xml":
        path = path.parent
    if not path.is_dir():
        raise ValueError(f"OLX path is not a course directory or archive: {path}")
    if (path / "course.xml").is_file():
        return path
    matches = list(path.rglob("course.xml"))
    if len(matches) == 1:
        return matches[0].parent
    raise ValueError(f"Could not find course.xml under {path}")


def _html_payload(root: Path, url_name: str) -> str:
    html_file = root / "html" / f"{url_name}.html"
    xml_file = root / "html" / f"{url_name}.xml"
    chunks: list[str] = []
    if html_file.is_file():
        markup = html_file.read_text(encoding="utf-8", errors="replace")
        chunks.append(markup)
        for match in _STATIC_SRC.findall(markup):
            static_path = root / "static" / match
            if static_path.is_file():
                chunks.append(static_path.read_text(encoding="utf-8", errors="replace"))
    if xml_file.is_file() and not chunks:
        # Some OLX stores body inline in the XML file.
        chunks.append(xml_file.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _load_vertical(root: Path, url_name: str) -> OlxBlock:
    path = root / "vertical" / f"{url_name}.xml"
    if not path.is_file():
        return OlxBlock(url_name, url_name, "vertical")
    element = _parse_xml(path)
    texts: list[str] = []
    for html_name in _child_url_names(element, "html"):
        texts.append(html_to_text(_html_payload(root, html_name)))
    text = "\n\n".join(part for part in texts if part)
    return OlxBlock(url_name, _attr(element, "display_name", url_name), "vertical", text)


def _load_sequential(root: Path, url_name: str) -> OlxBlock:
    path = root / "sequential" / f"{url_name}.xml"
    if not path.is_file():
        return OlxBlock(url_name, url_name, "sequential")
    element = _parse_xml(path)
    children = tuple(_load_vertical(root, name) for name in _child_url_names(element, "vertical"))
    text = "\n\n".join(child.text for child in children if child.text)
    return OlxBlock(url_name, _attr(element, "display_name", url_name), "sequential", text, children)


def _load_chapter(root: Path, url_name: str) -> OlxBlock:
    path = root / "chapter" / f"{url_name}.xml"
    if not path.is_file():
        return OlxBlock(url_name, url_name, "chapter")
    element = _parse_xml(path)
    children = tuple(_load_sequential(root, name) for name in _child_url_names(element, "sequential"))
    text = "\n\n".join(child.text for child in children if child.text)
    return OlxBlock(url_name, _attr(element, "display_name", url_name), "chapter", text, children)


def load_olx(path: str | Path) -> OlxCourse:
    root = _locate_course_dir(Path(path))
    course_xml = _parse_xml(root / "course.xml")
    org = _attr(course_xml, "org", "ORG")
    course = _attr(course_xml, "course", "COURSE")
    run = _attr(course_xml, "url_name", "run")
    display_name = f"{org} {course}"
    nested = root / "course" / f"{run}.xml"
    if nested.is_file():
        nested_root = _parse_xml(nested)
        display_name = _attr(nested_root, "display_name", display_name)
        chapter_names = _child_url_names(nested_root, "chapter")
    else:
        chapter_names = _child_url_names(course_xml, "chapter")
    chapters = tuple(_load_chapter(root, name) for name in chapter_names)
    return OlxCourse(org, course, run, display_name, root, chapters)


def format_outline(course: OlxCourse) -> str:
    lines = [f"{course.course_id}  {course.display_name}"]
    for chapter in course.chapters:
        lines.append(f"  {chapter.url_name}  {chapter.display_name}")
        for sequential in chapter.children:
            lines.append(f"    {sequential.url_name}  {sequential.display_name}")
            for vertical in sequential.children:
                lines.append(
                    f"      {vertical.url_name}  {vertical.display_name}  ({vertical.char_count} chars)"
                )
    return "\n".join(lines)


def _matches(block: OlxBlock, query: str) -> bool:
    needle = query.strip().lower()
    return needle in {block.url_name.lower(), block.display_name.lower()} or needle in block.url_name.lower() or needle in block.display_name.lower()


def _is_lesson_sequential(block: OlxBlock) -> bool:
    return not _SKIP_SEQUENTIAL.search(block.display_name or "")


def _is_lesson_vertical(block: OlxBlock) -> bool:
    return not _SKIP_VERTICAL.search(block.display_name or "")


def find_chapter(course: OlxCourse, query: str) -> OlxBlock:
    for chapter in course.chapters:
        if _matches(chapter, query):
            return chapter
    options = ", ".join(chapter.url_name for chapter in course.chapters) or "(none)"
    raise ValueError(f"No unit matched {query!r}. Known units: {options}")


def find_vertical(course: OlxCourse, query: str) -> tuple[OlxBlock, OlxBlock]:
    for chapter in course.chapters:
        for sequential in chapter.children:
            for vertical in sequential.children:
                if _matches(vertical, query):
                    return chapter, vertical
    raise ValueError(f"No section matched {query!r}. Use --list to see ids.")


def brief_for_unit(course: OlxCourse, unit_query: str) -> LessonBrief:
    chapter = find_chapter(course, unit_query)
    pages: list[OlxBlock] = []
    section_ids: list[str] = []
    for sequential in chapter.children:
        if not _is_lesson_sequential(sequential):
            continue
        for vertical in sequential.children:
            if not _is_lesson_vertical(vertical) or not vertical.text:
                continue
            pages.append(vertical)
            section_ids.append(vertical.url_name)
    if not pages:
        raise ValueError(f"Unit {chapter.url_name} has no lesson text to generate from.")
    source = "\n\n".join(f"## {page.display_name}\n{page.text}" for page in pages)
    return LessonBrief(
        title=f"{course.display_name}: {chapter.display_name}",
        course_id=course.course_id,
        unit_id=chapter.url_name,
        unit_title=chapter.display_name,
        section_ids=section_ids,
        source_text=source,
    )


def brief_for_section(course: OlxCourse, section_query: str) -> LessonBrief:
    chapter, vertical = find_vertical(course, section_query)
    if not vertical.text:
        raise ValueError(f"Section {vertical.url_name} has no extractable lesson text.")
    return LessonBrief(
        title=f"{course.display_name}: {vertical.display_name}",
        course_id=course.course_id,
        unit_id=chapter.url_name,
        unit_title=chapter.display_name,
        section_ids=[vertical.url_name],
        source_text=f"## {vertical.display_name}\n{vertical.text}",
    )
