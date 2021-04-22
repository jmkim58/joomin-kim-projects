import React from "react";
import ReactDOM from "react-dom";
import { PrifinaProvider, PrifinaContext, usePrifina } from "@prifina/hooks";
import Select from "react-select";
import Box from "@blend-ui/core/dist/esm/Box";
import { Text, Flex } from "@blend-ui/core";

// import ProgressContainer from "./ProgressContainer";
import ToDoList from "./ToDoList";
import ToDoList2 from "./ToDoList2";
import ToDoItems from "./ToDoItems";

import data from "./data/data.json";

import { deviceOptions, getDeviceData } from "../src/data/helper";
// import ProgressCard from "./ProgressCard"

import { BrowserRouter as Router, Route, Switch } from "react-router-dom";

import { ThemeProvider } from "@blend-ui/core";
import "./styles/styles";

// import { ThemeProvider } from "@blend-ui/core";
// import { extendTheme } from "@chakra-ui/react";


{/*
import {
  extendTheme,
  ThemeProvider,
  CircularProgress,
  CircularProgressLabel,
} from "@chakra-ui/react";
*/}


// import { usePrifina } from "@prifina/hooks";
// import { ThemeProvider } from "@blend-ui/core";

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

{/*
export const LocalComponent = (props) => {
  return (
    <PC {...props} />
  );
};*/}


var destination = document.querySelector("#container")
  

{/*
ReactDOM.render (
  <div>
      <ToDoList />
  </div>,
  destination
);
*/}




export const App = () => {
  return (
    <React.Fragment>
      <Router>
        <ThemeProvider>
          <ToDoList2 />
        </ThemeProvider>
      </Router>
    </React.Fragment>
  );
};



{/* WORKING 
export const App = () => {
  return (
    <React.Fragment>
      <Router>
        <ThemeProvider>
          <ToDoList />
        </ThemeProvider>
      </Router>
    </React.Fragment>
  );
};
*/}

// <Route exact path="/" exact component={PC} />


{/*
export const App = () => {
  return(
  <ThemeProvider theme={theme}>
      <Flex
        marginLeft={251}
        paddingLeft={70}
        paddingTop={55}
        justifyContent={"space-between"}
      >
        <Box>
          <Box marginBottom={75} paddingTop={0}>
            <Flex justifyContent={"space-between"}>
              <Flex marginBottom={0}>
                <Text fontSize={34} color="#5F6AC4">
                  Health Overview
                </Text>
              </Flex>
              <Flex paddingRight={35}>
                <Flex>
                  <Box>
                    <Flex minHeight="43px">
                      <div
                        style={{
                          minWidth: 180,
                          paddingRight: 35,
                        }}
                      >
                        <Select
                          defaultValue="time"
                          placeholder="Time"
                          styles={selectStyleShadow}
                          options={timeOptions}
                          onChange={handleTimeChange}
                        />
                      </div>
                      <div style={{ width: 100 }}>
                        <Select
                          defaultValue="Device"
                          placeholder="Device"
                          styles={selectStyleShadow}
                          options={deviceOptions}
                          onChange={handleDeviceChange}
                        />
                      </div>
                    </Flex>
                    <Flex paddingRight={300} paddingBottom={27}>
                      <Text fontSize={24} color="#5F6AC4"></Text>
                    </Flex>
                  </Box>
                </Flex>
              </Flex>
            </Flex>
            <Flex>
              <Flex
                width="634px"
                height="308px"
                borderRadius={10}
                style={styles}
                justifyContent="center"
                alignItems="center"
              >
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
              </Flex>
              {/*
              <Flex paddingLeft={40}>
                <Box paddingRight={20}>
                  <Card
                    icon={SleepIcon}
                    smallIcon={GreenArrowIcon}
                    title="7h 30m"
                    subtitle="sleep"
                    precentage="11%"
                    iconColor="#E9D8FD"
                    color="#FAF5FF"
                  />
                  <Box paddingTop={15}>
                    <Card
                      icon={HeartIcon}
                      smallIcon={RedArrowIcon}
                      title="89bpm"
                      subtitle="rate"
                      precentage="5%"
                      iconColor="#FED7D7"
                      color="#FFF5F5"
                    />
                  </Box>
                </Box>
                <Box>
                  <Card
                    icon={TemperatureIcon}
                    smallIcon={GreenArrowIcon}
                    title="36.6"
                    subtitle="body temp"
                    precentage="10%"
                    iconColor="#B2F5EA"
                    color="#E6FFFA"
                  />
                </Box>
              </Flex>
            </Flex>
          </Box>
          <Box>
            <Box marginBottom={30}>
              <Text fontSize={34} color="#5F6AC4">
                Monthly Goals
              </Text>
            </Box>
            <Flex>
              <Box height={299}>
                <ProgressCard
                  title={calories + " kcal"}
                  subtitle="out of 100 000"
                  color="#F6E05E"
                  value={calories / 1000}
                  icon={FireIcon}
                />
                <ProgressCard
                  title={(sleep / 60).toFixed(0) + " hours"}
                  subtitle="out of 240"
                  color="#4299E1"
                  value={sleep / 140}
                  icon={SleepIcon}
                />
              </Box>
              <Box height={299}>
                <ProgressCard
                  title={steps + " steps"}
                  subtitle="out of 600 000"
                  color="#ED8936"
                  value={steps / 6000}
                  icon={WalkIcon}
                />
                <ProgressCard
                  title={(sleep / 60).toFixed(2) + " hours"}
                  subtitle="out of 240"
                  color="#4299E1"
                  value={sleep / 240}
                  icon={SleepIcon}
                /> 
              </Box>
            </Flex>
            </Box>
                </Box> 
            </Flex>
          </Box>
        </Box>
        <Box marginRight={50}>
        </Flex>
        </Box>
        </Box>
      </Flex> 
    </ThemeProvider>
  );
}
*/}

 

export default App; 