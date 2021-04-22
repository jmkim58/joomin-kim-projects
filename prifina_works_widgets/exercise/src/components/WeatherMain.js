import React from "react";
import useFetch from "../hooks/UseFetch";
import { API_KEY, API_BASE_URL } from "../apis/config";
//import { Text, Avatar, Flex } from "@blend-ui/core";
import {Text, Flex} from "@chakra-ui/react";

const WeatherMain = (props) => {
  const city = "Toronto";  //Ottawa


  const { data, error, isLoading, setUrl } = useFetch(
    `${API_BASE_URL}/data/2.5/weather?q=${city}&units=metric&appid=${API_KEY}`
  );
  //api.openweathermap.org/data/2.5/weather?q={city name}&appid={API key}

  const getContent = () => {
    if (error) return <h2>Error when fetching: {error}</h2>;
    if (!data && isLoading) return <h2>LOADING...</h2>;
    if (!data) return null;
    console.log(JSON.stringify(data));

    const icon = data.weather[0].icon;
    return (
      <Flex
        marginLeft={10}
        marginRight={10}
        justifyContent={"flex-start"}
        display={'flex'}
      >

        <img color={"#E9EDF0"} 
          width={"27%"}
          height={"27%"}

          src={`http://openweathermap.org/img/wn/${icon}@2x.png`}
        />

        <Flex
          marginLeft={2}
          marginBottom={25}
          marginTop={15}


          justifyContent={"space-between"}
          flexDirection={"column"}>

          <Text
            style={{
              textDecoration: "none",
              fontWeight: 400,
              color: "#FFFFFF",
              fontSize: 21,
            }}>
            Forcast
            </Text>

          <div style={{
            textTransform: "capitalize",
            textDecoration: "none",
            fontWeight: 400,
            color: "#7fffd4",
            fontSize: 18.5,
          }}>

            {data.weather[0].description}
          </div>
        </Flex>

      </Flex>
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

export default WeatherMain;
