import React from "react";
import { Card } from "react-bootstrap";

const WeatherCard = ({ dt, temp_min, temp_max, main, icon, temp }) => {
  // create a date object with Date class constructor
  const date = new Date(dt);

  return (
    <Card>
      <Card.Img
        variant="top"
        // get the src from example url and pass the icon prop for icon code
        src={`http://openweathermap.org/img/wn/${icon}@2x.png`}
      />

      <Card.Body>
        <Card.Title
          style={{
            position: "relative",
            top: "-50px",
            left: "9px",
            backgroundColor: "white",
            width: "60px",
            borderRadius: "10px",
          }}
        >
          {temp}
          {"°C"}
        </Card.Title>
        <Card.Subtitle>{main}</Card.Subtitle>
        {/*  datetime is received in milliseconds, let's turn into local date time */}
        <p>
          {date.toLocaleDateString()} - {date.toLocaleTimeString()}
        </p>

        {/* minimum temperature */}
        <p>Min: {temp_min}</p>
        {/* maximum temperature */}
        <p>Max: {temp_max}</p>
      </Card.Body>
    </Card>
  );
};
export default WeatherCard;
