import re
import sys
import urllib.request
import modal


# app = modal.App(name="hello-world")


# @app.function()
# def get_links(url):
#     """
#     Get all links from a given URL.
#     """
#     response = urllib.request.urlopen(url)
#     html = response.read().decode("utf-8")  # Decode t
#     links = []
#     for match in re.finditer(r'href="(.*?)"', html):
#         links.append(match.group(1))
#     return links


# @app.local_entrypoint()
# def main():

#     links = get_links.remote("https://www.baidu.com")
#     print("Links found:")
#     for link in links:
#         print(link)


playright_image = modal.Image.debian_slim(python_version="3.10").run_commands(
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "pip install playwright==1.42.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)

import asyncio


# @app.function(image=playright_image)
async def get_links(url):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        links = await page.evaluate(
            """() => Array.from(document.querySelectorAll('a'), a => a.href)"""
        )
        await browser.close()
        print(f"Found {len(links)} links on {url}  {links}")
    return links


asyncio.run(get_links("https://www.baidu.com"))
