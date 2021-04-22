import React from "react";
import useFetch from "./hooks/useFetch";
import { API_KEY, API_BASE_URL } from "./apis/config";
//import WeatherList from "./WeatherList";
//import WeatherCard from "./WeatherCard";
//import { Container } from "react-bootstrap";

const App = (props) => {
  const city = "san francisco";
  /*
  const { data, error, isLoading, setUrl } = useFetch(
    `${API_BASE_URL}/data/2.5/forecast?q=${city}&cnt=5&units=metric&appid=${API_KEY}`
  );
*/

  const { data, error, isLoading, setUrl } = useFetch(
    `${API_BASE_URL}/data/2.5/weather?q=${city}&units=metric&appid=${API_KEY}`
  );
  //api.openweathermap.org/data/2.5/weather?q={city name}&appid={API key}

  const getContent = () => {
    if (error) return <h2>Error when fetching: {error}</h2>;
    if (!data && isLoading) return <h2>LOADING...</h2>;
    if (!data) return null;
    console.log(JSON.stringify(data));
    /*
    return <WeatherList weathers={data.list} />;
const WeatherCard = ({ dt, temp_min, temp_max, main, icon }) =>

    {"coord":{"lon":24.9355,"lat":60.1695},"weather":[{"id":802,"main":"Clouds","description":"scattered clouds",
    "icon":"03d"}],"base":"stations","main":{"temp":-5.81,"feels_like":-11.21,"temp_min":-6.11,
    "temp_max":-4.44,"pressure":1018,"humidity":86},"visibility":10000,"wind":{"speed":3.6,"deg":150},
    "clouds":{"all":40},"dt":1613732443,"sys":{"type":1,"id":1332,"country":"FI","sunrise":1613713683,
    "sunset":1613748027},"timezone":7200,"id":658225,"name":"Helsinki","cod":200}

    */
    /*
    return (
      <WeatherCard
        dt={data.dt * 1000}
        temp={data.main.temp}
        temp_min={data.main.temp_min}
        temp_max={data.main.temp_max}
        main={data.weather[0].description}
        icon={data.weather[0].icon}
      />
    );
    */
    const icon = data.weather[0].icon;
    return (
      <React.Fragment>
        <div
          style={{
            width: "200px",
            height: "200px",
            padding: "5px",
          }}
        >
          <div style={{ textAlign: "center", textTransform: "capitalize" }}>
            {data.weather[0].description}
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              justifyContent: "center",
              alignItems: "center",
              height: "100px",
            }}
          >
            <img
              width={"100"}
              src={`http://openweathermap.org/img/wn/${icon}@2x.png`}
            />
            <div>
              {Math.ceil(data.main.temp)}
              {"°C"}
            </div>
          </div>
          <div style={{ textAlign: "center", textTransform: "capitalize" }}>
            {city}
          </div>
        </div>
      </React.Fragment>
    );
  };

  return (
    <div>
      {/* 
        <CitySelector onSearch={(city) => setUrl(`${API_BASE_URL}/data/2.5/forecast?q=${city}&cnt=5&appid=${API_KEY}`)} />
  */}
      {/* conditionally render  */}
      {getContent()}
    </div>
  );
};

export default App;
