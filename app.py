from typing import List, Dict, Union, Any

from fastapi.middleware.cors import CORSMiddleware

import deepdoctection as dd
from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="DeepDoctection API")
origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> List[Dict[str, Union[Union[int, List[str], List[Any]], Any]]]:
    analyzer = dd.get_dd_analyzer()
    df = analyzer.analyze(path=file.file)
    df.reset_state()
    doc = iter(df)

    pages_result = []
    for i, page in enumerate(doc):
        content = []
        for layout in page.layouts:
            if layout.category_name == "title":
                content.append(f'title : {layout.text}')
                # title = layout.text
            elif layout.category_name == "list":
                content.append(f'list : {layout.text}')
                # list = layout.text
        text = page.text
        tables = page.tables or []
        tables_html = [table.html for table in tables]
        page_result = {
            "page_number": i + 1,
            "content": content,
            "text": text,
            "table": tables_html,
        }
        pages_result.append(page_result)
    return pages_result


def main():
    import uvicorn

    uvicorn.run("app:app", port=8000)


if __name__ == "__main__":
    main()
