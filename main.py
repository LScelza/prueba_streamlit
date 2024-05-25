import os
import uvicorn 
from fastapi import FastAPI, Request, Query, Path,HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title='Proyecto Integrador I Hecho por Michael Martinez')
# Cargar el dataset
df_recom = pd.read_parquet('Dataset/recomendacion3_v1.parquet')
# Preprocesamiento de los géneros para generar la matriz TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_recom.groupby('item_id')['genres'].apply(lambda x: ' '.join(x)))
# Cálculo de la similitud del coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
@app.get("/Sistema_de_recomendacion")
async def recomendacion_juego(item_id : float = Query(default=22330.0)):
    try:
        # Obtener el índice del juego en el DataFrame
        idx = df_recom[df_recom['item_id'] == item_id].index[0]
        # print('Existe')
    except IndexError:
        # print('No existe')
        # Manejar el caso donde no se encuentra el ID del juego
        raise HTTPException(status_code=404, detail="Item ID no encontrado.")
    # Calcular la similitud de coseno entre el juego dado y todos los juegos
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Ordenar los juegos por su similitud de coseno
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    # Obtener los índices de los juegos recomendados
    game_indices = [i[0] for i in sim_scores]
    # Obtener los títulos de los juegos recomendados
    juegos_recomendados = df_recom['title'].iloc[game_indices].tolist() 
    return juegos_recomendados

# if __name__ == "__main__":
#     port = int(os.environ.get('PORT', 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)


