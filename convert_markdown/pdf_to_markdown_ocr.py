from pyzerox import zerox
import os
import asyncio
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()

load_dotenv()

model = "gpt-4o-2024-11-20" 

async def main():
    for file_path in os.listdir("./documents_with_english_titles"):
        file_path = f"./documents_with_english_titles/{file_path}"

        kwargs = {}
        custom_system_prompt = None
        select_pages = None
        output_dir = "./output" 
        result = await zerox(
            file_path=file_path,
            model=model,
            output_dir=output_dir,
            custom_system_prompt=custom_system_prompt,
            select_pages=select_pages,
            **kwargs,
        )
        print(f"{file_path} is done")
    return result

result = asyncio.run(main())