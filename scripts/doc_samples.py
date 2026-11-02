"""Small made-up documents, one per file type the chat's document reader accepts, each holding one fact nobody could guess.
Used by scripts/chat_race.py --docs: the chat reads each file, then must answer a question about its fact.
samples() returns [(file name, mime type, bytes, question, word the answer must contain)].
"""
import io
import zipfile


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, body, stored in files:
            if stored:
                z.writestr(zipfile.ZipInfo(name), body, compress_type=zipfile.ZIP_STORED)
            else:
                z.writestr(name, body)
    return buf.getvalue()


def docx(text):
    return _zip([
        ("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>', False),
        ("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>', False),
        ("word/document.xml", f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>', False)])


def xlsx(text):
    return _zip([
        ("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>', False),
        ("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>', False),
        ("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>', False),
        ("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>', False),
        ("xl/worksheets/sheet1.xml", f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Fact</t></is></c><c r="B1" t="inlineStr"><is><t>{text}</t></is></c></row></sheetData></worksheet>', False)])


def pptx(text):
    return _zip([
        ("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/></Types>', False),
        ("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>', False),
        ("ppt/presentation.xml", '<?xml version="1.0" encoding="UTF-8"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>', False),
        ("ppt/_rels/presentation.xml.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>', False),
        ("ppt/slides/slide1.xml", f'<?xml version="1.0" encoding="UTF-8"?><p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>', False)])


def odt(text):
    return _zip([
        ("mimetype", "application/vnd.oasis.opendocument.text", True),
        ("META-INF/manifest.xml", '<?xml version="1.0" encoding="UTF-8"?><manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2"><manifest:file-entry manifest:full-path="/" manifest:media-type="application/vnd.oasis.opendocument.text"/><manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/></manifest:manifest>', False),
        ("content.xml", f'<?xml version="1.0" encoding="UTF-8"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" office:version="1.2"><office:body><office:text><text:p>{text}</text:p></office:text></office:body></office:document-content>', False)])


def epub(text):
    return _zip([
        ("mimetype", "application/epub+zip", True),
        ("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>', False),
        ("content.opf", '<?xml version="1.0" encoding="UTF-8"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Log</dc:title><dc:identifier id="id">x1</dc:identifier><dc:language>en</dc:language></metadata><manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/></spine></package>', False),
        ("c1.xhtml", f'<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Log</title></head><body><p>{text}</p></body></html>', False)])


def pdf(text):
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"]
    stream = f"BT /F1 14 Tf 72 700 Td ({text}) Tj ET"
    objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offs = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out.encode()))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    x = len(out.encode())
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n"
    return out.encode()


def samples():
    return [
        ("keeper.txt", "text/plain", b"The lighthouse keeper's cat is called Biscuit. Biscuit sleeps on the third step of the tower.", "What is the lighthouse keeper's cat called?", "biscuit"),
        ("tide.csv", "text/csv", b"place,fact\nQuay,The tide table is posted by the Quay clerk named Ferris\n", "Who posts the tide table?", "ferris"),
        ("notes.md", "text/markdown", b"# Notes\n\nThe fog horn sounds twice every minute: a long note, then a short one.\n", "How often does the fog horn sound?", "twice"),
        ("page.html", "text/html", b"<html><body><h1>Harbour</h1><p>The pilot boat is painted bright yellow.</p></body></html>", "What colour is the pilot boat?", "yellow"),
        ("harbour.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", docx("The harbour master is called Odile and she rings a green bell at dawn."), "What is the harbour master called?", "odile"),
        ("ferry.pdf", "application/pdf", pdf("The ferry to Marrow Island leaves at seven forty every morning."), "When does the ferry to Marrow Island leave?", "seven forty"),
        ("lamp.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", xlsx("The signal lamp burns paraffin oil"), "What does the signal lamp burn?", "paraffin"),
        ("rota.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", pptx("The rescue boat crew is led by Captain Thessaly"), "Who leads the rescue boat crew?", "thessaly"),
        ("chart.odt", "application/vnd.oasis.opendocument.text", odt("The sea chart was drawn by a mapmaker named Quillon"), "Who drew the sea chart?", "quillon"),
        ("log.epub", "application/epub+zip", epub("The ship's log records a storm on Tuesday night"), "When was the storm in the ship's log?", "tuesday"),
        ("mail.mbox", "application/mbox", b"From sender@example.com Mon Jan  1 00:00:00 2024\nFrom: sender@example.com\nTo: keeper@example.com\nSubject: Delivery\n\nThe supply boat brings forty crates of salt on Friday.\n", "How many crates of salt does the supply boat bring?", "forty"),
    ]
