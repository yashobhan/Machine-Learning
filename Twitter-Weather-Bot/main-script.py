import tweepy, requests, time
from pytz import timezone
from datetime import datetime


CONSUMER_KEY = ''
CONSUMER_SECRET = ''
ACCESS_KEY = ''
ACCESS_SECRET = ''
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
api = tweepy.API(auth)


while(True):
    url = 'http://api.openweathermap.org/data/2.5/weather?appid=YOURAPIKEY&q=Mumbai'
    poll_url = 'https://api.waqi.info/feed/geo:19;72/?token=YOURTOkEN'

    json_data = requests.get(url).json()
    poll_json_data = requests.get(poll_url).json()

    condition = json_data['weather'][0]['main']
    temperature = json_data['main']['temp']
    humidity = json_data['main']['humidity']
    windspeed = json_data['wind']['speed']

    pollution = poll_json_data['data']['aqi']

    temperature = temperature - 273.15
    windspeed = windspeed * 3.6

    degree_sign = u'\N{DEGREE SIGN}'


    india = timezone('Asia/Kolkata')
    currenttime = datetime.now(india).strftime('%H:%M on %d-%m-%Y')


    print("running..")
    tweet = "It's " + str(currenttime) + ", the weather is " + str(round(temperature)) + degree_sign + " Celsius with "\
            + str(condition) + ". Humidity at " + str(humidity) + "%. " + "Wind speed is " + str(round(windspeed, 1)) \
            + "km/hr. " + "Air Quality Index: " + str(pollution) + "\n#Mumbai #Weather #Forecast #Climate #Bombay"

    print(tweet)
    api.update_status(status=tweet)

    time.sleep(7200)
