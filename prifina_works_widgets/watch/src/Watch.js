import React, { useRef, useEffect, useState } from "react";
import styled from "styled-components";

/*
$primary: black;
$primary-2: white;
$secondary: gray;
$accent: red;
$clock-border: 10px;
*/
const Container = styled.div`
  padding: 10px;
  height: 200px;
  font-size: 10px;
  width: 200px;
`;
const testProps = (props) => {
  console.log("PROPS ", props);
  return null;
};
const Clock = styled.div`
  /*background-color: mix(white, gray, 95%); */
  border-radius: 50%;
  width: ${(props) => props.size};
  height: ${(props) => props.size};
  display: flex;
  position: relative;
  box-shadow: inset 0 0 20px gray;
  border: 4px solid black;

  .hours-hand {
    box-shadow: 1px 1px 1px 1px gray;
    border-radius: 40% 40% 20% 20%;
    position: absolute;
    border: 2px solid black;
    background-color: white;
    width: 3px;
  }

  .minutes-hand {
    box-shadow: 1px 1px 1px 1px gray;
    position: absolute;
    border-radius: 40% 40% 20% 20%;
    background-color: white;
    border: 1px solid black;
    width: 2px;
  }

  .seconds-hand {
    box-shadow: 1px 1px 1px 0px red;
    position: absolute;
    border-radius: 30%;
    background-color: red;
    width: 1px;
  }

  .hand-pivot {
    box-shadow: 0 0 2px 2px red;
    border: 3px solid white;
    border-radius: 50%;
    height: 0px;
    width: 0px;
    position: absolute;
  }

  .dial-hour {
    /*color: mix(white, black, 30%); */
    text-shadow: 1px 1px 4px;
    position: absolute;
  }
  .dial-hour.hour-main {
    color: black;
    font-size: 1.6em;
    font-weight: 500;
  }
`;
const Hour = styled.span`
  top: ${(props) => (props.pos ? props.pos.top : 0)}px;
  left: ${(props) => (props.pos ? props.pos.left : 0)}px;
`;
const Pivot = styled.span`
  top: ${(props) => (props.pos ? props.pos.top : 0)}px;
  left: ${(props) => (props.pos ? props.pos.left : 0)}px;
`;
const HoursHand = styled.div`
  position: absolute;
  top: ${(props) => (props.pos ? props.pos.top : 0)}px;
  left: ${(props) => (props.pos ? props.pos.left : 0)}px;
  height: ${(props) => (props.pos ? props.pos.height : 0)}px;
  transform-origin: ${(props) => (props.pos ? props.pos.origin : 0)};
`;
const MinutesHand = styled.div`
  top: ${(props) => (props.pos ? props.pos.top : 0)}px;
  left: ${(props) => (props.pos ? props.pos.left : 0)}px;
  height: ${(props) => (props.pos ? props.pos.height : 0)}px;
  transform-origin: ${(props) => (props.pos ? props.pos.origin : 0)}px;
`;
const SecondsHand = styled.div`
  top: ${(props) => (props.pos ? props.pos.top : 0)}px;
  left: ${(props) => (props.pos ? props.pos.left : 0)}px;
  height: ${(props) => (props.pos ? props.pos.height : 0)}px;
  transform-origin: ${(props) => (props.pos ? props.pos.origin : 0)}px;
`;
const getDims = (elm, removeBorder = true, removePadding = true) => {
  const elmStyles = window.getComputedStyle(elm);
  return {
    height:
      elm.offsetHeight +
      (removeBorder
        ? -parseInt(elmStyles.borderTopWidth, 10) -
          parseInt(elmStyles.borderBottomWidth, 10)
        : 0) +
      (removePadding
        ? -parseInt(elmStyles.paddingTop, 10) -
          parseInt(elmStyles.paddingBottom, 10)
        : 0),
    width:
      elm.offsetWidth +
      (removeBorder
        ? -parseInt(elmStyles.borderLeftWidth, 10) -
          parseInt(elmStyles.borderRightWidth, 10)
        : 0) +
      (removePadding
        ? -parseInt(elmStyles.paddingLeft, 10) -
          parseInt(elmStyles.paddingRight, 10)
        : 0),
  };
};

const clockwiseRotate = (center, angle, point) => {
  const movex = point.x - center.x;
  const movey = point.y - center.y;

  const s = Math.sin((angle * Math.PI) / 180);
  const c = Math.cos((angle * Math.PI) / 180);

  const x = movex * c - movey * s;
  const y = movex * s + movey * c;

  return {
    x: x + center.x,
    y: y + center.y,
  };
};

function useIsMountedRef() {
  const isMountedRef = useRef(null);
  useEffect(() => {
    isMountedRef.current = true;
    return () => (isMountedRef.current = false);
  });
  return isMountedRef;
}

const Watch = () => {
  const isMountedRef = useIsMountedRef();
  const [clockSize, setClockSize] = useState("100%");
  const [dialHour, setDialHour] = useState([]);
  const [handPivot, setHandPivot] = useState({ top: 0, left: 0 });
  const [hoursHand, setHoursHand] = useState({
    top: 0,
    left: 0,
    height: 0,
    origin: 0,
  });
  const [minutesHand, setMinutesHand] = useState({
    top: 0,
    left: 0,
    height: 0,
    origin: 0,
  });
  const [secondsHand, setSecondsHand] = useState({
    top: 0,
    left: 0,
    height: 0,
    origin: 0,
  });

  useEffect(() => {
    let timeoutId = null;
    let intervalId = null;
    if (isMountedRef.current) {
      const clock = document.getElementById("clock");
      //console.log("MOUNTED ", clock);

      const { height: boxH, width: boxW } = getDims(clock, false, false);
      const minv = Math.min(boxH, boxW);
      //clock.style.height = clock.style.width = minv
      setClockSize(minv + "px");

      const { height, width } = getDims(clock);

      let dialHours = document.getElementsByClassName("dial-hour");
      let offsetFix = dialHours[11].offsetHeight / 10;
      let refx = 0;
      let refy = -height / 2 + dialHours[11].offsetHeight / 2;
      const origin = { x: 0, y: 0 };
      let _dialHours = dialHour;
      for (let i = 1; i <= 12; ++i) {
        let newc = clockwiseRotate(origin, 30, { x: refx, y: refy });
        refx = Math.round(newc.x);
        refy = Math.round(newc.y);
        newc.x = Math.round(
          newc.x - dialHours[i - 1].offsetWidth / 2 + width / 2
        );
        newc.y = Math.round(
          newc.y - dialHours[i - 1].offsetHeight / 2 + height / 2 + offsetFix
        );
        _dialHours.push({ top: newc.y, left: newc.x });
        //dialHours[i - 1].style.top = newc.y + 'px'
        //dialHours[i - 1].style.left = newc.x + 'px'
      }
      //console.log("DIAL ", _dialHours);
      setDialHour(_dialHours);

      const handPivotElement = document.getElementById("hand-pivot");
      const pivotBoxDims = getDims(handPivotElement, false, false);
      setHandPivot({
        top: height / 2 - pivotBoxDims.height / 2,
        left: width / 2 - pivotBoxDims.width / 2,
      });

      const hoursHandElement = document.getElementById("hours-hand");

      const minutesHandElement = document.getElementById("minutes-hand");
      const secondsHandElement = document.getElementById("seconds-hand");
      const offByPivot = 0.05 * height;

      const hoursHandDims = getDims(hoursHandElement, false, false);

      const minutesHandDims = getDims(minutesHandElement, false, false);
      const secondsHandDims = getDims(secondsHandElement, false, false);

      const hoursHandHeight = height / 2 - 1.6 * dialHours[11].offsetHeight;
      //height / 2 - 1.6 * dialHours[11].offsetHeight + offByPivot,
      setHoursHand({
        top: 1.6 * dialHours[11].offsetHeight,
        left: width / 2 - hoursHandDims.width / 2,
        height: hoursHandHeight + offByPivot,
        origin: `${hoursHandDims.width / 2}px ${hoursHandHeight}px`,
      });

      const minutesHandHeight = height / 2 - 1.2 * dialHours[11].offsetHeight;
      //height / 2 - 1.2 * dialHours[11].offsetHeight + offByPivot,
      setMinutesHand({
        top: 1.2 * dialHours[11].offsetHeight,
        left: width / 2 - minutesHandDims.width / 2,
        height: minutesHandHeight + offByPivot,
        origin: `${minutesHandDims.width / 2}px ${minutesHandHeight}`,
      });
      const secondsHandHeight = height / 2 - dialHours[11].offsetHeight;
      // height / 2 - dialHours[11].offsetHeight + offByPivot,
      setSecondsHand({
        top: dialHours[11].offsetHeight,
        left: width / 2 - secondsHandDims.width / 2,
        height: secondsHandHeight + offByPivot,
        origin: `${secondsHandDims.width / 2}px ${secondsHandHeight}`,
      });

      const dt = new Date();
      const secsElpased = dt.getSeconds();
      const minsElapsed = dt.getMinutes() + secsElpased / 60;
      const hrsElapsed = (dt.getHours() % 12) + minsElapsed / 60;

      const rotate = (elm, deg) => {
        elm.style.transform = `rotate(${deg}deg)`;
      };

      let hrsRotn = (hrsElapsed * 360) / 12;
      let minsRotn = (minsElapsed * 360) / 60;
      let secsRotn = (secsElpased * 360) / 60;
      rotate(hoursHandElement, hrsRotn);
      rotate(minutesHandElement, minsRotn);
      rotate(secondsHandElement, secsRotn);

      timeoutId = setTimeout(
        () => (secondsHandElement.style.transition = "transform 1s linear"),
        0
      );
      intervalId = setInterval(() => {
        // do %360
        hrsRotn += 360 / (3600 * 12);
        minsRotn += 360 / (60 * 60);
        secsRotn += 360 / 60;
        rotate(hoursHandElement, hrsRotn);
        rotate(minutesHandElement, minsRotn);
        rotate(secondsHandElement, secsRotn);
      }, 1000);
    }
    return () => {
      clearTimeout(timeoutId);
      clearInterval(intervalId);
    };
  }, [isMountedRef]);

  return (
    <Container>
      <Clock id="clock" size={clockSize}>
        <HoursHand
          pos={hoursHand}
          className="hours-hand"
          id="hours-hand"
        ></HoursHand>
        <MinutesHand
          pos={minutesHand}
          className="minutes-hand"
          id="minutes-hand"
        ></MinutesHand>
        <SecondsHand
          pos={secondsHand}
          className="seconds-hand"
          id="seconds-hand"
        ></SecondsHand>
        <Pivot pos={handPivot} className="hand-pivot" id="hand-pivot"></Pivot>
        {[...Array(12)].map((x, i) => {
          if ((i + 1) % 3 === 0) {
            return (
              <Hour
                pos={dialHour[i]}
                key={"hour-" + i}
                className={"dial-hour hour-main"}
              >
                {i + 1}
              </Hour>
            );
          } else {
            return (
              <Hour pos={dialHour[i]} key={"hour-" + i} className={"dial-hour"}>
                {i + 1}
              </Hour>
            );
          }
        })}
      </Clock>
    </Container>
  );
};

export default Watch;
