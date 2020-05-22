import requests
import pandas as pd
import datetime
import numpy as np
import itertools
import os
from sklearn.externals import joblib
from app.config import Config


darkskykey = Config['darkskykey']
googlekey = Config['googlekey']


script_path = os.path.dirname(os.path.abspath( __file__ + "/../"))
accident_dataset = pd.read_csv(script_path + "/data/only_accident_points.csv")


model = joblib.load(script_path + "/model/model.pkl")
model_columns = joblib.load(script_path + "/model/model_columns.pkl")


def call_google(origin, destination, googlekey):
    PARAMS = {'origin': origin, 'destination': destination, 'key': googlekey, }
    URL = "https://maps.googleapis.com/maps/api/directions/json"
    res = requests.get(url=URL, params=PARAMS)
    data = res.json()

    waypoints = data['routes'][0]['legs']

    lats = []
    longs = []
    google_count_lat_long = 0

    for leg in waypoints:
        for step in leg['steps']:
            start_loc = step['start_location']
            lats.append(start_loc['lat'])
            longs.append(start_loc['lng'])
            google_count_lat_long += 1

    lats = tuple(lats)
    longs = tuple(longs)
    print("total waypoints: " + str(google_count_lat_long))

    return lats, longs, google_count_lat_long



def calc_distance(accident_dataset, lats, longs, google_count_lat_long):
    accident_point_counts = len(accident_dataset.index)

    R = 6373.0
    new = accident_dataset.append([accident_dataset] * (google_count_lat_long - 1), ignore_index=True)  # repeat data frame (9746*waypoints_count) times
    lats_r = list(
        itertools.chain.from_iterable(itertools.repeat(x, accident_point_counts) for x in lats))  # repeat 9746 times
    longs_r = list(itertools.chain.from_iterable(itertools.repeat(x, accident_point_counts) for x in longs))

    new['lat2'] = np.radians(lats_r)
    new['long2'] = np.radians(longs_r)

    new['lat1'] = np.radians(new['Latitude'])
    new['long1'] = np.radians(new['Longitude'])
    new['dlon'] = new['long2'] - new['long1']
    new['dlat'] = new['lat2'] - new['lat1']

    new['a'] = np.sin(new['dlat'] / 2) ** 2 + np.cos(new['lat1']) * np.cos(new['lat2']) * np.sin(new['dlon'] / 2) ** 2
    new['distance'] = R * (2 * np.arctan2(np.sqrt(new['a']), np.sqrt(1 - new['a'])))

    return new



def call_darksky(clusters, darkskykey, tm):
    weather = pd.DataFrame()
    print(tm)

    datetime_str = datetime.datetime.strptime(tm, '%Y-%m-%dT%H:%M').strftime('%Y-%m-%dT%H:%M')
    tm2 = datetime_str[0:10] + "T" + datetime_str[11:13] + ":00:00"

    for index, row in clusters.iterrows():
        lat = row["Latitude"]
        long = row["Longitude"]
        weather_url = "https://api.darksky.net/forecast/" + darkskykey + "/" + str(lat) + "," + str(long) + "," + tm2 + \
                      "?exclude=[currently,minutely,daily,flags]"
        w_response = requests.get(weather_url)
        w_data = w_response.json()

        datetime_object = datetime.datetime.strptime(tm, '%Y-%m-%dT%H:%M')
        iweather = pd.DataFrame(w_data["hourly"]["data"][datetime_object.hour], index=[0])
        iweather["Cluster"] = row['Cluster']
        iweather['precipAccumulation'] = 0

        weather = weather.append(iweather)

    return weather



def model_pred(new_df):

    prob = pd.DataFrame(model.predict_proba(new_df), columns=['No', 'probability'])
    prob = prob[['probability']]
    output = prob.merge(new_df[['Latitude', 'Longitude']], how='outer', left_index=True, right_index=True)

    output["Latitude"] = round(output["Latitude"], 5)
    output["Longitude"] = round(output["Longitude"], 5)
    output = output.drop_duplicates(subset=['Longitude', 'Latitude'], keep="last")

    processed_results = []
    for index, row in output.iterrows():
        lat = float(row['Latitude'])
        long = float(row['Longitude'])
        prob = float(row['probability'])

        result = {'lat': lat, 'lng': long, 'probability': prob}
        processed_results.append(result)

    print("total accident count:", len(output))

    return processed_results



def api_call(origin, destination, tm):

    datetime_object = datetime.datetime.strptime(tm, '%Y-%m-%dT%H:%M')

    lats, longs, google_count_lat_long = call_google(origin, destination, googlekey)

    dist = calc_distance(accident_dataset, lats, longs, google_count_lat_long)

    dat = dist[dist['distance'] < 0.050][['Longitude','Latitude','Day_of_Week','Local_Authority_(District)',
                                               '1st_Road_Class','1st_Road_Number','Speed_limit', 'Year','Cluster',
                                               'Day_of_year', 'Hour']]

    if len(dat) == 0:
        return print(" Hooray! No accidents predicted in your route.")

    else:
      
        dat = dat.drop(columns=['Hour', 'Day_of_year', 'Day_of_Week', 'Year'], axis=0)
        dat['Hour'] = datetime_object.hour
        day_of_year = (datetime_object - datetime.datetime(datetime_object.year, 1, 1)).days + 1
        dat['Day_of_year'] = day_of_year
        day_of_week = datetime_object.date().weekday() + 1
        dat['Day_of_Week'] = day_of_week
        dat['Year'] = datetime_object.year

        ucluster = list(dat['Cluster'].unique())
        clusters = dat[dat['Cluster'].isin(ucluster)].drop_duplicates(subset='Cluster', keep='first')
        weather = call_darksky(clusters, darkskykey, tm)

        final_df = pd.merge(dat, weather, how='left', on=['Cluster'])
        final_df = final_df.drop(columns=['time', 'summary', 'icon', 'ozone'], axis=0)
        final_df = final_df[model_columns]

        processed_results = model_pred(final_df)

        final = {}
        final["accidents"] = processed_results

        return final

