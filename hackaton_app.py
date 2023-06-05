
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.express as px
from PIL import Image
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import dash
from streamlit_searchbox import st_searchbox
from streamlit_elements import elements, mui, html, sync
import enum
import logging
import random
import time
from typing import List, Tuple
import plotly.graph_objects as go
import requests
from streamlit_searchbox import st_searchbox

# Configuration de la page et caches

st.set_page_config(layout="wide",
    page_title = "Data Blue Notes",
    page_icon = "✨",

)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

@st.cache_data
def load_data():
    df = pd.read_csv('df_final.csv.gz', compression='gzip')
    df.drop(columns=['Unnamed: 0'], inplace=True)  # Drop the Unnamed: 0 column
    return df
df_final = load_data()

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


logging.getLogger("streamlit_searchbox").setLevel(logging.DEBUG)

# Barre latérale

logo = Image.open('DBN_presentation.png')

with st.sidebar:
    st.image(logo)
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    selected = option_menu (None, ['✨ Présentation ✨', '📈 KPI 📈', '🎶 Recommandations 🎶'], 
                         icons=['✨', '✨', '✨'],    
                          menu_icon="cast", default_index=0, 
                          styles={
        "container": {"padding": "0!important", "background-color": "#F2F4F5", "text-align": "center"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "18px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#434770"},
         })
selected

# Diapo présentation

if selected == '✨ Présentation ✨':

    IMAGES = [
        "https://www.obs-ed.fr/wp-content/uploads/2023/05/1.jpg",
        "https://www.obs-ed.fr/wp-content/uploads/2023/05/2-1.jpg",
        "https://www.obs-ed.fr/wp-content/uploads/2023/05/3-1.jpg",
        "https://www.obs-ed.fr/wp-content/uploads/2023/05/4-1.jpg",
        "https://www.obs-ed.fr/wp-content/uploads/2023/05/5.jpg",
        "https://www.obs-ed.fr/wp-content/uploads/2023/05/6-1.jpg", 
        "https://www.obs-ed.fr/wp-content/uploads/2023/05/7.jpg"
    ]

    def slideshow_swipeable(images):
        # Generate a session state key based on images.
        key = f"slideshow_swipeable_{str(images).encode().hex()}"

        # Initialize the default slideshow index.
        if key not in st.session_state:
            st.session_state[key] = 0

        # Get the current slideshow index.
        index = st.session_state[key]

        # Create a new elements frame.
        with elements(f"frame_{key}"):

            # Use mui.Stack to vertically display the slideshow and the pagination centered.
            # https://mui.com/material-ui/react-stack/#usage
            with mui.Stack(spacing=2, alignItems="center"):

                # Create a swipeable view that updates st.session_state[key] thanks to sync().
                # It also sets the index so that changing the pagination (see below) will also
                # update the swipeable view.
                # https://mui.com/material-ui/react-tabs/#full-width
                # https://react-swipeable-views.com/demos/demos/
                with mui.SwipeableViews(index=index, resistance=True, onChangeIndex=sync(key)):
                    for image in images:
                        html.img(src=image, css={"width": "100%"})

                # Create a handler for mui.Pagination.
                # https://mui.com/material-ui/react-pagination/#controlled-pagination
                def handle_change(event, value):
                    # Pagination starts at 1, but our index starts at 0, explaining the '-1'.
                    st.session_state[key] = value-1

                # Display the pagination.
                # As the index value can also be updated by the swipeable view, we explicitely
                # set the page value to index+1 (page value starts at 1).
                # https://mui.com/material-ui/react-pagination/#controlled-pagination
                mui.Pagination(page=index+1, count=len(images), color="primary", onChange=handle_change)


    if __name__ == '__main__':

    
        slideshow_swipeable(IMAGES)    


# KPI

elif selected == '📈 KPI 📈':

# Pour faire fonctionner certains graphiques

    musiques_per_decade = df_final.value_counts('decade')
    musiques_per_decade = musiques_per_decade.to_frame().reset_index()
    musiques_per_decade = musiques_per_decade.rename(columns={0: 'nombre_musiques'})
    musiques_per_decade = musiques_per_decade.sort_values('decade')

    evolution_tempo = df_final.groupby(['decade'])['tempo'].mean()
    evolution_tempo = evolution_tempo.to_frame().reset_index()

    evolution_danceability = df_final.groupby(['decade'])['danceability'].mean()    
    evolution_danceability = evolution_danceability.to_frame().reset_index()

    lg_morceaux_tps = df_final.groupby(['decade'])['duration_ms'].mean()
    lg_morceaux_tps_test = lg_morceaux_tps.to_frame().reset_index()

    f_comptage_explicit = df_final.groupby(['explicit', 'decade']).size().reset_index(name='COUNT')
    f_comptage_explicit = f_comptage_explicit.loc[f_comptage_explicit['explicit']==True]

    valence_decade = df_final.groupby(['decade'])['valence'].mean()
    tab_valence_decade = valence_decade.to_frame().reset_index()

    variation = tab_valence_decade["valence"].pct_change()

    tab_valence_decade["variation"] = variation
    tab_valence_decade = tab_valence_decade.drop(tab_valence_decade.index[0])
    
# Onglet

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["   Caractéristiques  ",
                                            "   Décennies   ", 
                                            "   Décennies bis   ",
                                            "   Popularité    ",
                                            "   Ternaire   "])

#Onglet 1 : Heatmap
    
    with tab1:
            plt.figure(figsize=(11,5))
            df_cor =df_final[['explicit', 'danceability', 'energy',
                'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
                'time_signature', 'popularity']]
            cmap = sns.diverging_palette(220, 20, center='light', as_cmap=True)
            mat_correl = sns.heatmap(df_cor.corr(), vmin=-1, vmax=1, cmap=cmap, annot=True)
            plt.title('Correlation entres les principales carcatéristiques \n')#vmin et vmax pour centrer les couleurs sur 0
            st.pyplot(mat_correl.get_figure())
            
#Onglet 2 : décennies
    
    with tab2:
        st.header("Graphiques par décennie")
        col1, col2 = st.columns(2, gap="large")

        with col1:
                plotA = px.bar(musiques_per_decade, x='decade', y='count', 
                labels={'decade': 'Décennie', 'nombre_musiques': 'Nombre de musiques'})
                plotA.update_layout(
                    title='Nombre de morceaux par décennie',
                    xaxis_title='Décennie',
                    yaxis_title='Nombre de morceaux',
                )
                st.plotly_chart(plotA, use_column_width='auto')

               
        with col2:
                plotB = px.line(lg_morceaux_tps_test, x='decade', y='duration_ms')

                plotB .update_layout(
                    title='Durée moyenne des morceaux par décennie',
                    xaxis_title='Décennie',
                    yaxis_title='Durée (ms)',
                )

                st.plotly_chart(plotB , use_container_width=True)

# Onglet 3 : décennies

    with tab3:
        st.header("Graphiques par décennie")
        col1, col2 = st.columns(2, gap="large")

        with col1:

                plotA2, ax1a = plt.subplots(figsize=(7,5.5))
                ax1a.plot(evolution_tempo['decade'], evolution_tempo['tempo'], color='tab:red')
                ax1a.set_xlabel('Décennie')
                ax1a.set_ylabel('Tempo', color='tab:red')

                ax2a = ax1a.twinx()
                ax2a.plot(evolution_danceability['decade'], evolution_danceability['danceability'], color='tab:blue')
                ax2a.set_ylabel('Danceability', color='tab:blue')

                plt.title('Evolution du tempo et de danceability par décennie')
                st.pyplot(plotA2)

        with col2:

                plotB2, ax1b = plt.subplots(figsize=(7,6))
                ax1b.bar(f_comptage_explicit['decade'], f_comptage_explicit['COUNT'], color='tab:grey')
                ax1b.set_xlabel('Décennie')
                ax1b.set_ylabel('Morceaux à paroles explicites', color='tab:grey')
                ax2b = ax1b.twinx()
                ax2b.plot(tab_valence_decade['decade'], tab_valence_decade['variation'], color='tab:red')
                ax2b.set_ylabel('Positivité', color='tab:red')
                ax2b.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
                plt.title('Evolution de la positivité et des paroles explicites par décennie')
                st.pyplot(plotB2)

# Onglet 4 avec API de Spotify       
                

    with tab4 :        
        col1, col2 = st.columns([3, 1], gap="large")
        with col1:

                Types_of_Features = ("acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence")

                st.header('Mesures avec la popularité')

                Name_of_Artist = st.text_input("Nom de l'artiste")
                Name_of_Feat = st.selectbox("Features", Types_of_Features)
                button_clicked = st.button("OK")

                from spotipy_client import *
            

                client_id = '0fb92619fee741a8a40acd3d2a15a628'
                client_secret = 'c262eac04c954700aabb437530f0af82'

                spotify = SpotifyAPI(client_id, client_secret)

                    
                Data = spotify.search({"artist": f"{Name_of_Artist}"}, search_type="track")

                need = []
                for i, item in enumerate(Data['tracks']['items']):
                    track = item['album']
                    track_id = item['id']
                    song_name = item['name']
                    popularity = item['popularity']
                    need.append((i, track['artists'][0]['name'], track['name'], track_id, song_name, track['release_date'], popularity))
                    
                Track_df = pd.DataFrame(need, index=None, columns=('Item', 'Artist', 'Album Name', 'Id', 'Song Name', 'Release Date', 'Popularity'))

                access_token = spotify.access_token

                headers = {
                        "Authorization": f"Bearer {access_token}"
                }
                endpoint = "https://api.spotify.com/v1/audio-features/"

                Feat_df = pd.DataFrame()
                for id in Track_df['Id'].iteritems():
                    track_id = id[1]
                    lookup_url = f"{endpoint}{track_id}"
                    ra = requests.get(lookup_url, headers=headers)
                    audio_feat = ra.json()
                    Features_df = pd.DataFrame(audio_feat, index=[0])
                    Feat_df = Feat_df.append(Features_df)

                Full_Data = Track_df.merge(Feat_df, left_on="Id", right_on="id")

                Sort_DF = Full_Data.sort_values(by=['Popularity'], ascending=False)

                chart_df = Sort_DF[['Artist', 'Album Name', 'Song Name', 'Release Date', 'Popularity', f'{Name_of_Feat}']]

                import altair as alt

                feat_header = Name_of_Feat.capitalize()    

                st.header(f'{feat_header}' " vs. Popularity")
                c = alt.Chart(chart_df).mark_circle().encode(
                    alt.X('Popularity', scale=alt.Scale(zero=False)), y=f'{Name_of_Feat}', color=alt.Color('Popularity', scale=alt.Scale(zero=False)), 
                    size=alt.value(200), tooltip=['Popularity', f'{Name_of_Feat}', 'Song Name', 'Album Name'])

                st.altair_chart(c, use_container_width=True)

                st.header("Classement avec les chansons les plus populaires")
                st.table(chart_df)

        with col2:
                st.header("Features")
                st.write("Acousticness: Mesure sur le caractère acoustique d'une piste. La valeur est entre 0 et 1.")
                st.divider()
                st.write("Danceability: Mesure si un morceau se prête à la danse sur la base d'une combinaison d'éléments musicaux tels que le tempo, la stabilité du rythme, la force de la pulsation et la régularité générale. Une valeur de 0,0 est la moins dansante et une valeur de 1,0 est la plus dansante")
                st.divider()
                st.write("Energy: L'énergie est une mesure comprise entre 0,0 et 1,0 et représente une mesure perceptive de l'intensité et de l'activité.Les caractéristiques perceptives contribuant à cet attribut comprennent la gamme dynamique, l'intensité sonore perçue, le timbre, la vitesse d'apparition et l'entropie générale.")
                st.divider()
                st.write("Instrumentalness: Détermine si une piste ne contient pas de voix. Plus la valeur de l'instrumentalité est proche de 1,0, plus il est probable que la piste ne contienne pas de contenu vocal.")
                st.divider()
                st.write("Liveness: Détecte la présence d'un public dans l'enregistrement. ")
                st.divider()
                st.write("Loudness: L'intensité sonore globale d'une piste en décibels (dB). Les valeurs d'intensité sonore sont calculées en moyenne sur l'ensemble de la piste et sont utiles pour comparer l'intensité sonore relative des pistes. ")
                st.divider()
                st.write("Speechiness: L'aptitude à la parole détecte la présence de mots parlés dans une piste.")
                st.divider()
                st.write("Tempo: Le tempo global estimé d'une piste en battements par minute (BPM)")
                st.divider()
                st.write("Valence:Une mesure de 0,0 à 1,0 décrivant la positivité musicale véhiculée par une piste. Les pistes ayant une valence élevée sonnent plus positivement (par exemple, joyeux, gai, euphorique), tandis que les pistes ayant une valence faible sonnent plus négativement (par exemple, triste, déprimé, en colère).")
                st.divider()
                st.divider()
                st.write("Information :  https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/")

# Onglet 5 : ternaire

    with tab5 :        
        col1, col2 = st.columns([3, 1], gap="large")
        with col1:                
            Types_of_Features = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence"]

            with st.form(key='my_form'):
                Name_of_Artist_2 = st.text_input("Nom de l'artiste", key='artist_name')
                Feat_1 = st.selectbox("Feature 1", Types_of_Features, index=0, key='feat_1')
                Feat_2 = st.selectbox("Feature 2", Types_of_Features, index=1, key='feat_2')
                Feat_3 = st.selectbox("Feature 3", Types_of_Features, index=2, key='feat_3')
                button_clicked = st.form_submit_button("OK")

                if button_clicked:
                    client_id = '0fb92619fee741a8a40acd3d2a15a628'
                    client_secret = 'c262eac04c954700aabb437530f0af82'
                    spotify = SpotifyAPI(client_id, client_secret)
                    Data = spotify.search({"artist": f"{Name_of_Artist_2}"}, search_type="track")
                    need = []
                    for i, item in enumerate(Data['tracks']['items']):
                        track = item['album']
                        track_id = item['id']
                        song_name = item['name']
                        popularity = item['popularity']
                        need.append((i, track['artists'][0]['name'], track['name'], track_id, song_name, track['release_date'], popularity))
                    Track_df = pd.DataFrame(need, index=None, columns=('Item', 'Artist', 'Album Name', 'Id', 'Song Name', 'Release Date', 'Popularity'))
                    access_token = spotify.access_token
                    headers = {"Authorization": f"Bearer {access_token}"}
                    endpoint = "https://api.spotify.com/v1/audio-features/"
                    Feat_df = pd.DataFrame()
                    for id in Track_df['Id'].iteritems():
                        track_id = id[1]
                        lookup_url = f"{endpoint}{track_id}"
                        ra = requests.get(lookup_url, headers=headers)
                        audio_feat = ra.json()
                        Features_df = pd.DataFrame(audio_feat, index=[0])
                        Feat_df = Feat_df.append(Features_df)
                    Full_Data = Track_df.merge(Feat_df, left_on="Id", right_on="id")
                    Sort_DF = Full_Data.sort_values(by=['Popularity'], ascending=False)

                    chart_df = Sort_DF[['Artist', 'Album Name', 'Song Name', 'Release Date', 'Popularity', Feat_1, Feat_2, Feat_3]]
                    fig = px.scatter_ternary(chart_df, a=Feat_1, b=Feat_2, c=Feat_3, color='Popularity', size='Popularity', hover_data=['Song Name', 'Album Name', 'Popularity'])
                    st.plotly_chart(fig, use_container_width=True)
                    st.header("Classement avec les chansons les plus populaires")
                    st.table(chart_df)

        with col2:
                st.header("Features")
                st.write("Acousticness: Mesure sur le caractère acoustique d'une piste. La valeur est entre 0 et 1.")
                st.divider()
                st.write("Danceability: Mesure si un morceau se prête à la danse sur la base d'une combinaison d'éléments musicaux tels que le tempo, la stabilité du rythme, la force de la pulsation et la régularité générale. Une valeur de 0,0 est la moins dansante et une valeur de 1,0 est la plus dansante")
                st.divider()
                st.write("Energy: L'énergie est une mesure comprise entre 0,0 et 1,0 et représente une mesure perceptive de l'intensité et de l'activité.Les caractéristiques perceptives contribuant à cet attribut comprennent la gamme dynamique, l'intensité sonore perçue, le timbre, la vitesse d'apparition et l'entropie générale.")
                st.divider()
                st.write("Instrumentalness: Détermine si une piste ne contient pas de voix. Plus la valeur de l'instrumentalité est proche de 1,0, plus il est probable que la piste ne contienne pas de contenu vocal.")
                st.divider()
                st.write("Liveness: Détecte la présence d'un public dans l'enregistrement. ")
                st.divider()
                st.write("Loudness: L'intensité sonore globale d'une piste en décibels (dB). Les valeurs d'intensité sonore sont calculées en moyenne sur l'ensemble de la piste et sont utiles pour comparer l'intensité sonore relative des pistes. ")
                st.divider()
                st.write("Speechiness: L'aptitude à la parole détecte la présence de mots parlés dans une piste.")
                st.divider()
                st.write("Tempo: Le tempo global estimé d'une piste en battements par minute (BPM)")
                st.divider()
                st.write("Valence:Une mesure de 0,0 à 1,0 décrivant la positivité musicale véhiculée par une piste. Les pistes ayant une valence élevée sonnent plus positivement (par exemple, joyeux, gai, euphorique), tandis que les pistes ayant une valence faible sonnent plus négativement (par exemple, triste, déprimé, en colère).")
                st.divider()
                st.divider()
                st.write("Information :  https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/")
        

# Système de recommendation

elif selected == '🎶 Recommandations 🎶':

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from streamlit_searchbox import st_searchbox


    X = df_final[['explicit', 'danceability', 'energy', 'loudness', 'mode', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                'key_0', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7',
                'key_8', 'key_9', 'key_10', 'key_11']]

    distanceKNN = NearestNeighbors(n_neighbors=4).fit(X)

    def search_sth_fast(searchterm: str) -> List[str]:
        if not searchterm:
            return []
        try:
            result = df_final.loc[df_final['name'].str.contains(searchterm, case=False, na=False), ['name', 'artists']] \
                .sort_values(by='popularity', ascending=False).values
            formatted_result = [f"{name_tuple[0]} par {name_tuple[1][1:-1]}" for name_tuple in result]
            return formatted_result
        except KeyError:
            return []

    st.write ("N.B. Les recommandations prennent en compte les caractériques musicales d'une chanson.")
    st.write()
    st.write ("Rentrez le nom d'une chanson")
    # Search box
    selected_values2 = st_searchbox(
        search_sth_fast,
        default=None,
        clear_on_submit=True,
        key='song_names'
    )


    def get_recommendations(selected_song, selected_artist):
        song = df_final.loc[df_final['name'] == selected_song]
        song = song.loc[song['artists'].apply(lambda x: selected_artist in x)]

        if song.empty:
            st.error(f"{selected_song} par {selected_artist} n'a pas été trouvé .. désolé !")
            return -1, [], [], [], []

        stats = song[X.columns]
        distances, indices = distanceKNN.kneighbors(stats)

        selected_song_id = song.iloc[0]['id']
        selected_song_index = indices[0][0]  # Index of the selected song in the recommendations

        # Exclude the selected song from the recommendations
        song_names = df_final.iloc[indices[0][1:]]['name'].tolist()
        artist_names = df_final.iloc[indices[0][1:]]['artists'].tolist()
        preview_urls = df_final.iloc[indices[0][1:]]['preview_url'].tolist()
        ids = df_final.iloc[indices[0][1:]]['id'].tolist()

        return selected_song_index, song_names, artist_names, preview_urls, ids


    if selected_values2:
        my_song = selected_values2.split(" par ")[0]
        my_artist = selected_values2.split(" par ")[1]
        selected_song_index, song_names, artist_names, preview_urls, ids = get_recommendations(my_song, my_artist)
        st.header(f"3 recommendations liées à {my_song} par {my_artist}")
        st.divider()

        for i in range(3):
            st.header(f'{song_names[i]}')
            st.subheader(f'Par {artist_names[i][1:-1]}')
            st.write('')
            preview = (f'{preview_urls[i]}')
            st.audio(preview, format='audio/mp3')
            url = 'https://open.spotify.com/track/' + ids[i]
            st.write("🎵 [Spotify](%s)" % url)

        # Additional information based on selected song index
        if selected_song_index != -1:
            st.write(f"La chanson selectionnées '{my_song}' par {my_artist} est à l'index {selected_song_index} des recommendations")
