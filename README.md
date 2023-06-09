# Nederlandse trainingsdata voor AI-modellen

In deze repository vind je de code voor ons onderzoek naar het Nederlandstalige deel van Google's MC4 – de grootste collectie aan Nederlandstalige websites die beschikbaar is om AI-modellen te trainen. 

Het belangrijkste deel van het onderzoek is te vinden in de map `notebooks`. In `download_and_process_mc4.ipynb` downloaden we de dumps van de [MC4-*repository* op *Hugging Face*](https://huggingface.co/datasets/mc4). Omdat de totale dataset meer dan 100GB omvat, verwerken we die dumps één voor één, waarbij we domeinnaam extraheren en het aantal woorden tellen. Ook voegen we de websitelabels toe, die verkregen zijn met behulp van [Homepage2Vec](https://github.com/epfl-dlab/homepage2vec). We veronderstellen dat de websitelabels al bekend zijn. Wanneer je zelf het model wil draaien, kan je gebruik maken van `fetch_websites.py` (om de websites op te halen) en `classify_websites.py` (om een label toe te kennen aan opgehaalde websites), die je tegelijkertijd moet laten draaien. De scripts downloaden maximaal 10.000 websites tegelijkertijd. Op een RTX 3080 nam het classificeren van alle websites in de dataset iets meer dan drie dagen in beslag.

De analyse en de visualisatie van de data is te vinden in `analysis_and_visualization.ipynb`. We maken de visualisaties direct drukklaar – dus daar wordt een groot deel van de code aan besteed. 
