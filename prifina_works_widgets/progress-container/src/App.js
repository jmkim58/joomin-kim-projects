import React from "react";
import { PrifinaProvider, PrifinaContext } from "@prifina/hooks";
import ProgressContainer from "./ProgressContainer";

import { BrowserRouter as Router, Route, Switch } from "react-router-dom";

import { usePrifina } from "@prifina/hooks";

import { ThemeProvider } from "@blend-ui/core";

/*
// this is only for local webpack server test  => yarn start
export const LocalComponent = (props) => {
  return (
    <PrifinaProvider stage={"dev"} Context={PrifinaContext}>
      <Hello {...props} />
    </PrifinaProvider>
  );
};
*/

export const LocalComponent = (props) => {
  return (
    <PC {...props} />
  );
};





/*
<ProgressContainer
                  text="Week 1"
                  value1={barCalories1}
                  value2={barSteps1}
                  value3={barDistance1}
                  value4={barSleep1}
                  value5={value5}
                  value6={value6}
                  value7={value7}
                  hoverValue6={barSleep1}
                />
                <ProgressContainer
                  text="Week 2"
                  value1={barCalories2}
                  value2={barSteps2}
                  value3={barDistance2}
                  value4={barSleep2}
                  value5={value5}
                  value6={value6}
                  value7={value7}
                />
                <ProgressContainer
                  text="Week 3"
                  value1={barCalories3}
                  value2={barSteps3}
                  value3={barDistance3}
                  value4={barSleep3}
                  value5={value5}
                  value6={value6}
                  value7={value7}
                />
                <ProgressContainer
                  text="Week 4"
                  value1={barCalories4}
                  value2={barSteps4}
                  value3={barDistance4}
                  value4={barSleep4}
                  value5={value5}
                  value6={value6}
                  value7={value7}
                />
*/ 