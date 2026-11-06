"""Read an Open edX OLX export and expose its ordered teaching content."""

from __future__ import annotations

import io
import base64
import json
import re
import html as html_module
import tarfile
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET


MAX_ARCHIVE_BYTES = 250 * 1024 * 1024
MAX_FILES = 20_000
MAX_COMPONENT_CHARS = 40_000
MAX_COURSE_CHARS = 1_500_000
MAX_UNIT_CHARS = 50_000


class _IframeSourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "iframe":
            source = dict(attrs).get("src")
            if source:
                self.sources.append(source)


@dataclass
class Component:
    kind: str
    title: str
    text: str


@dataclass
class Unit:
    id: str
    title: str
    components: list[Component] = field(default_factory=list)


@dataclass
class UnitSelection:
    section_title: str
    subsection_title: str
    unit: Unit
    component_indices: list[int]


@dataclass
class Subsection:
    id: str
    title: str
    graded: bool
    units: list[Unit] = field(default_factory=list)


@dataclass
class Section:
    id: str
    title: str
    subsections: list[Subsection] = field(default_factory=list)


@dataclass
class Course:
    title: str
    sections: list[Section]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"script", "style", "noscript", "iframe"}:
            self.hidden_depth += 1
        if tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")
        if tag == "img" and attributes.get("alt"):
            self.parts.append(attributes["alt"] or "")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "iframe"} and self.hidden_depth:
            self.hidden_depth -= 1
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def html_to_text(source: str) -> str:
    parser = _TextExtractor()
    parser.feed(source)
    text = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _xblock_markup(root: ET.Element) -> str:
    """Get escaped HTML from the serialized XBlock without losing nested markup."""
    text = root.text or ""
    if text.strip():
        return html_module.unescape(text)
    return "".join(ET.tostring(child, encoding="unicode", method="html") for child in list(root))


class OLXReader:
    """Resolve Studio export references from course/run/chapters/sequentials/verticals."""

    def __init__(self, root: Path):
        self.root = root
        self._char_count = 0

    def _xml(self, path: Path) -> ET.Element:
        if not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"OLX is missing a required XML file or it is too large: {path.name}")
        try:
            return ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid OLX XML in {path.name}: {exc}") from exc

    @staticmethod
    def _path(base: Path, folder: str, ref: str) -> Path:
        name = PurePosixPath(ref).name
        if not name or name in {".", ".."}:
            raise ValueError("OLX contains an empty or unsafe component reference.")
        return base / folder / f"{name}.xml"

    @staticmethod
    def _component_path(base: Path, folder: str, ref: str) -> Path | None:
        name = PurePosixPath(ref).name
        if not name or name in {".", ".."}:
            raise ValueError("OLX contains an empty or unsafe component reference.")
        candidates = [base / folder / f"{name}.xml", base / f"{name}.xml"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        matches = list(base.glob(f"**/{name}.xml"))
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _component_file(base: Path, folder: str, ref: str, suffix: str) -> Path | None:
        name = PurePosixPath(ref).name
        if not name or name in {".", ".."}:
            raise ValueError("OLX contains an empty or unsafe component reference.")
        candidates = [base / folder / f"{name}{suffix}", base / f"{name}{suffix}"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        matches = list(base.glob(f"**/{name}{suffix}"))
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _static_iframe_page(base: Path, sources: list[str]) -> Path | None:
        for source in sources:
            if source.startswith("/") and not source.startswith("/static/"):
                continue
            reference = PurePosixPath(source.lstrip("/"))
            if ".." in reference.parts:
                continue
            parts = list(reference.parts)
            if parts and parts[0] == "static":
                parts = parts[1:]
            if not parts:
                continue
            candidate = base / "static" / Path(*parts)
            if candidate.is_file() and candidate.stat().st_size <= 4 * MAX_COMPONENT_CHARS:
                return candidate
        return None

    @staticmethod
    def _title(node: ET.Element, fallback: str) -> str:
        return (node.get("display_name") or node.get("name") or fallback).strip()

    def _contents(self, node: ET.Element, kind: str, fallback: str, base: Path) -> list[Component]:
        components: list[Component] = []
        for child in list(node):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in {"done", "course_info", "conditional", "prerequisite"}:
                continue
            title = self._title(child, tag.replace("_", " ").title())
            text = ""
            if tag == "html":
                filename = child.get("filename") or child.get("url_name")
                if filename:
                    html_file = self._component_file(base, "html", filename, ".html")
                    html_path = self._component_path(base, "html", filename)
                    if html_file:
                        wrapper = html_file.read_text(encoding="utf-8", errors="replace")
                        iframe_parser = _IframeSourceParser()
                        iframe_parser.feed(wrapper)
                        full_page = self._static_iframe_page(base, iframe_parser.sources)
                        text = html_to_text(full_page.read_text(encoding="utf-8", errors="replace")) if full_page else html_to_text(wrapper)
                    elif html_path:
                        html_node = self._xml(html_path)
                        text = html_to_text(_xblock_markup(html_node))
                    else:
                        text = html_to_text(_xblock_markup(child))
                else:
                    text = html_to_text(_xblock_markup(child))
                # Some OLX exports store a rich interactive lesson in a
                # sibling static HTML page and leave only an iframe wrapper
                # in the component file. Resolve that iframe for both .html
                # and .xml component variants.
                if filename and text:
                    iframe_parser = _IframeSourceParser()
                    iframe_parser.feed(text)
                    full_page = self._static_iframe_page(base, iframe_parser.sources)
                    if full_page and full_page.stat().st_size <= MAX_COMPONENT_CHARS * 4:
                        embedded_text = html_to_text(full_page.read_text(encoding="utf-8", errors="replace"))
                        if embedded_text:
                            text = embedded_text
            elif tag == "video":
                text = " ".join(filter(None, [child.get("youtube_id_1_0"), child.get("youtube_id_0_0"), child.get("url_name")]))
                text = f"Video reference: {text}" if text else "Video component (transcript may be in an associated asset)."
            elif tag == "problem":
                ref = child.get("url_name") or child.get("filename")
                problem_node = child
                if ref:
                    problem_path = self._component_path(base, "problem", ref)
                    if problem_path:
                        problem_node = self._xml(problem_path)
                text = html_to_text(_xblock_markup(problem_node))
            elif tag == "html5":
                filename = child.get("filename") or child.get("url_name")
                asset = (base / "static" / "html5" / PurePosixPath(filename).name / "index.html") if filename else None
                if asset and asset.is_file() and asset.stat().st_size <= MAX_COMPONENT_CHARS * 4:
                    text = html_to_text(asset.read_text(encoding="utf-8", errors="replace"))
            else:
                text = html_to_text(" ".join([child.get("url_name", ""), "".join(child.itertext())]))
            if text:
                text = text[:MAX_COMPONENT_CHARS]
                self._char_count += len(text)
                if self._char_count > MAX_COURSE_CHARS:
                    raise ValueError("The OLX course has too much extracted text (limit is 1.5 million characters).")
                components.append(Component(kind=tag, title=title, text=text))
        return components

    def read(self) -> Course:
        course_file = self.root / "course.xml"
        if not course_file.exists():
            raise ValueError("This OLX export has no course.xml. Upload an Open edX course export, not a learner-data archive.")
        course_ref = self._xml(course_file).get("url_name")
        if course_ref:
            run_file = self._path(self.root, "course", course_ref)
        else:
            candidates = sorted((self.root / "course").glob("*.xml")) if (self.root / "course").exists() else []
            if len(candidates) != 1:
                raise ValueError("Could not identify the course outline: course.xml must reference one course/<name>.xml file.")
            run_file = candidates[0]
        run_node = self._xml(run_file)
        sections: list[Section] = []
        for chapter_ref in _children(run_node, "chapter"):
            ref = chapter_ref.get("url_name")
            chapter = self._xml(self._path(self.root, "chapter", ref)) if ref else chapter_ref
            section = Section(ref or f"section-{len(sections)+1}", self._title(chapter, "Untitled section"))
            for seq_ref in _children(chapter, "sequential"):
                seq_id = seq_ref.get("url_name") or f"subsection-{len(section.subsections)+1}"
                sequential = self._xml(self._path(self.root, "sequential", seq_id)) if seq_ref.get("url_name") else seq_ref
                subsection = Subsection(seq_id, self._title(sequential, "Untitled subsection"), sequential.get("graded", "false").lower() == "true")
                vertical_nodes = _children(sequential, "vertical")
                if not vertical_nodes:
                    vertical_nodes = [sequential]
                for vertical_ref in vertical_nodes:
                    unit_id = vertical_ref.get("url_name") or f"unit-{len(subsection.units)+1}"
                    vertical = self._xml(self._path(self.root, "vertical", unit_id)) if vertical_ref.get("url_name") else vertical_ref
                    component_root = vertical
                    # Components may be nested in-line in OLX or referenced from component folders.
                    comps = self._contents(component_root, "unit", unit_id, self.root)
                    if comps:
                        unit_chars = sum(len(component.text) for component in comps)
                        if unit_chars > MAX_UNIT_CHARS:
                            raise ValueError(
                                f"OLX unit '{self._title(vertical, unit_id)}' contains {unit_chars:,} characters. "
                                f"The safe per-unit limit is {MAX_UNIT_CHARS:,}; split the source into smaller Open edX units."
                            )
                        subsection.units.append(Unit(unit_id, self._title(vertical, "Untitled unit"), comps))
                if subsection.units:
                    section.subsections.append(subsection)
            if section.subsections:
                sections.append(section)
        if not sections:
            raise ValueError("No text-bearing course units were found in this OLX export. Check that it contains chapter, sequential, and vertical content.")
        title = self._title(run_node, self._title(self._xml(course_file), "Imported Open edX course"))
        return Course(title=title, sections=sections)


def _safe_extract(upload_name: str, data: bytes, destination: Path) -> Path:
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("OLX archive exceeds the 250 MB upload limit.")
    name = upload_name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_FILES:
                raise ValueError("OLX archive contains too many files (limit is 20,000).")
            total = sum(item.file_size for item in entries)
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Uncompressed OLX archive exceeds the 250 MB limit.")
            for item in entries:
                p = PurePosixPath(item.filename)
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("OLX archive contains an unsafe path.")
            archive.extractall(destination)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > MAX_FILES or sum(max(0, member.size) for member in members) > MAX_ARCHIVE_BYTES:
                raise ValueError("OLX archive exceeds its file-count or uncompressed-size limit.")
            for member in members:
                p = PurePosixPath(member.name)
                if p.is_absolute() or ".." in p.parts or not (member.isfile() or member.isdir()):
                    raise ValueError("OLX archive contains an unsafe path or unsupported link.")
            archive.extractall(destination, members=members, filter="data")
    elif name.endswith(".xml"):
        (destination / "course.xml").write_bytes(data)
    else:
        raise ValueError("Upload a .tar.gz, .tgz, or .zip Open edX OLX course export.")
    # Studio exports can wrap the OLX root in a single top-level directory.
    if not (destination / "course.xml").exists():
        roots = [path.parent for path in destination.rglob("course.xml")]
        if len(roots) == 1:
            return roots[0]
    elif not (destination / "course").exists() and (destination / "course" / "course.xml").exists():
        return destination / "course"
    return destination


def load_course(upload_name: str, data: bytes, temp_parent: Path | None = None) -> tuple[Course, tempfile.TemporaryDirectory[str]]:
    temporary = tempfile.TemporaryDirectory(prefix="olx-import-", dir=temp_parent)
    try:
        root = _safe_extract(upload_name, data, Path(temporary.name))
        return OLXReader(root).read(), temporary
    except Exception:
        temporary.cleanup()
        raise


def load_course_folder(files: list[dict[str, str]], temp_parent: Path | None = None) -> tuple[Course, tempfile.TemporaryDirectory[str]]:
    """Rebuild a browser-selected unzipped OLX directory in a confined temp root."""
    temporary = tempfile.TemporaryDirectory(prefix="olx-folder-", dir=temp_parent)
    root = Path(temporary.name)
    total_bytes = 0
    try:
        for item in files:
            relative = item.get("path", "")
            encoded = item.get("content", "")
            path = PurePosixPath(relative)
            if not relative or path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("Selected folder contains an unsafe file path.")
            try:
                data = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise ValueError(f"Could not read selected file {relative!r}.") from exc
            total_bytes += len(data)
            if total_bytes > MAX_ARCHIVE_BYTES:
                raise ValueError("Selected OLX folder exceeds the 250 MB upload limit.")
            destination = root.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        course_root = root
        if not (course_root / "course.xml").exists():
            roots = [path.parent for path in root.rglob("course.xml")]
            if len(roots) != 1:
                raise ValueError("Select the unzipped Open edX course folder containing course.xml.")
            course_root = roots[0]
        return OLXReader(course_root).read(), temporary
    except Exception:
        temporary.cleanup()
        raise


def course_manifest(course: Course) -> dict[str, Any]:
    sections = []
    for section in course.sections:
        subsections = []
        for subsection in section.subsections:
            units = []
            for unit in subsection.units:
                units.append({"id": unit.id, "title": unit.title, "components": [{"id": str(index), "kind": c.kind, "title": c.title, "characters": len(c.text), "preview": c.text[:240]} for index, c in enumerate(unit.components)]})
            subsections.append({"id": subsection.id, "title": subsection.title, "graded": subsection.graded, "units": units})
        sections.append({"id": section.id, "title": section.title, "subsections": subsections})
    return {"title": course.title, "sections": sections}


def selected_units(course: Course, scope: str, section_id: str | None, subsection_id: str | None, unit_id: str | None) -> list[tuple[str, str, Unit]]:
    result: list[tuple[str, str, Unit]] = []
    for section in course.sections:
        if scope != "grade" and section.id != section_id:
            continue
        for subsection in section.subsections:
            if scope in {"subsection", "unit"} and subsection.id != subsection_id:
                continue
            for unit in subsection.units:
                if scope == "unit" and unit.id != unit_id:
                    continue
                result.append((section.title, subsection.title, unit))
    if scope not in {"grade", "section", "subsection", "unit"}:
        raise ValueError("Choose grade, section, subsection, or unit scope.")
    if not result:
        raise ValueError("The selected OLX scope contains no importable teaching units.")
    return result


def selected_subsection_pairs(course: Course, selections: list[tuple[str, str]]) -> list[tuple[str, str, Unit]]:
    """Select subsections by parent-section pair; OLX IDs can repeat across sections."""
    sections_by_id = {section.id: section for section in course.sections}
    result: list[tuple[str, str, Unit]] = []
    for section_id, subsection_id in dict.fromkeys(selections):
        section = sections_by_id.get(section_id)
        if section is None:
            raise ValueError(f"Unknown section selection: {section_id}")
        subsection = next((item for item in section.subsections if item.id == subsection_id), None)
        if subsection is None:
            raise ValueError(f"Unknown subsection selection in section {section.title!r}: {subsection_id}")
        result.extend((section.title, subsection.title, unit) for unit in subsection.units)
    if not result:
        raise ValueError("The selected OLX scope contains no importable teaching units.")
    return result


def selected_unit_components(course: Course, selections: list[dict[str, Any]]) -> list[UnitSelection]:
    """Resolve chosen OLX units and the specific components checked in each unit."""
    sections = {section.id: section for section in course.sections}
    result: list[UnitSelection] = []
    for selection in selections:
        section = sections.get(str(selection.get("section_id", "")))
        if not section:
            raise ValueError("A selected unit refers to an unknown section.")
        subsection = next((item for item in section.subsections if item.id == str(selection.get("subsection_id", ""))), None)
        if not subsection:
            raise ValueError(f"A selected unit is not in section {section.title!r}.")
        unit = next((item for item in subsection.units if item.id == str(selection.get("unit_id", ""))), None)
        if not unit:
            raise ValueError(f"A selected unit does not exist in subsection {subsection.title!r}.")
        raw_indices = selection.get("component_indices")
        if not isinstance(raw_indices, list) or not raw_indices:
            raise ValueError(f"Select at least one content subsection in unit {unit.title!r}.")
        try:
            indices = sorted(set(int(index) for index in raw_indices))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Selected content indices must be integers in unit {unit.title!r}.") from exc
        if any(index < 0 or index >= len(unit.components) for index in indices):
            raise ValueError(f"A selected content subsection does not exist in unit {unit.title!r}.")
        result.append(UnitSelection(section.title, subsection.title, unit, indices))
    if not result:
        raise ValueError("Select at least one unit to generate.")
    return result


def write_manifest(path: Path, course: Course) -> None:
    path.write_text(json.dumps(course_manifest(course), ensure_ascii=False, indent=2), encoding="utf-8")


def _children(node: ET.Element, local_name: str) -> list[ET.Element]:
    return [child for child in list(node) if child.tag.rsplit("}", 1)[-1] == local_name]
