import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

df = pd.DataFrame(columns=[
    "Année-Modèle", "Boite de vitesses", "Type de carburant", "Kilométrage",
    "Marque", "Modèle", "Nombre de portes", "Origine", "Première main",
    "Puissance fiscale", "État", "Prix"
])

base_url = "https://www.avito.ma/fr/maroc/voitures_d_occasion-%C3%A0_vendre?o="
num_pages = 500
save_every = 10

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

target_fields = [
    "Année-Modèle", "Boite de vitesses", "Type de carburant", "Kilométrage",
    "Marque", "Modèle", "Nombre de portes", "Origine", "Première main",
    "Puissance fiscale", "État"
]


for page in range(1, num_pages + 1):
    url = base_url + str(page)
    print(f"Traitement de la page {page}...")

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Échec page {page} : {e}")
        continue

    soup = BeautifulSoup(response.text, 'html.parser')
    container = soup.find('div', class_='sc-1nre5ec-1 crKvIr listing')

    if not container:
        print("Annonces non trouvées sur cette page.")
        continue

    annonces_urls = container.find_all("a", class_="sc-1jge648-0 jZXrfL")

    for annonce in annonces_urls:
        lien = annonce.get("href")

        try:
            annonce_resp = requests.get(lien, headers=HEADERS, timeout=10)
            annonce_resp.raise_for_status()
        except Exception as e:
            print(f"Échec annonce : {e}")
            continue

        annonce_soup = BeautifulSoup(annonce_resp.text, 'html.parser')
        annonce_data = {}

        details = annonce_soup.find("div", class_="sc-19cngu6-0 dnArJl")
        if not details:
            continue

        for div in details.find_all("div", recursive=False):
            key_span = div.find("span", class_="sc-1x0vz2r-0 bXFCIH")
            value_span = div.find("span", class_="sc-1x0vz2r-0 fjZBup")
            if key_span and value_span:
                key = key_span.text.strip()
                value = value_span.text.strip()
                annonce_data[key] = value

        row = [annonce_data.get(cle, "None") for cle in target_fields]

        price_tag = annonce_soup.find("p", class_="sc-1x0vz2r-0 lnEFFR sc-1veij0r-10 jdRkSM")
        price = "None"
        if price_tag:
            price = price_tag.text.replace('\u202f', '').replace('\xa0', '').replace('DH', '').strip().replace(' ', '')
        row.append(price)

        if len(row) == len(df.columns):
            df.loc[len(df)] = row

    if page % save_every == 0:
        df.to_csv("voitures_avito_tmp.csv", index=False, encoding="utf-8-sig")
        print(f"Sauvegarde temporaire à la page {page}")

    time.sleep(0.5)

df.to_csv("voitures_avito.csv", index=False, encoding="utf-8-sig")
print("Export CSV terminé : voitures_avito.csv")