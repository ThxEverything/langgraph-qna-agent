from PyPDF2 import PdfReader
from pathlib import Path
import re


def extract_articles_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # 정규표현식으로 "제n조 (제목)" 형태 탐색
    pattern = r"(제\s*\d+\s*조)(.*?)\n"
    matches = list(re.finditer(pattern, text, re.DOTALL))

    articles = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        article_title = match.group(1).replace(" ", "") + match.group(2).strip()
        articles.append((article_title, content))

    return articles


def save_articles_as_markdown(articles, output_path):
    output_path = Path(output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for title, content in articles:
            f.write(f"## {title}\n\n{content}\n\n")


if __name__ == "__main__":
    pdf_path = "civil_law/civil_law_20432_20250131.pdf"
    output_md = "output/civil_law_articles.md"

    articles = extract_articles_from_pdf(pdf_path)
    save_articles_as_markdown(articles, output_md)
    print(f"Markdown 저장 완료: {output_md}")
