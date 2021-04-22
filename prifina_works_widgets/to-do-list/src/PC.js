import React, { useState, useEffect } from "react";

import { Text, Flex } from "@blend-ui/core";
import Select from "react-select";

import Box from "@blend-ui/core/dist/esm/Box";

import ProgressContainer from "../src/ProgressContainer";
import ProgressCard from "../src/ProgressCard";
import Card from "./components/Card";


import { extendTheme, ThemeProvider } from "@chakra-ui/react";

{/*
import {
  extendTheme,
  ThemeProvider,
  CircularProgress,
  CircularProgressLabel,
} from "@chakra-ui/react";
*/}


import { deviceOptions, getDeviceData } from "../src/data/helper";

import { selectStyleShadow } from "../src/styles/styles";


const timeOptions = [{ label: "This Month", value: "thisMonth" }];

const styles = {
  boxShadow: " 0px 5px 20px #F0F4F8",
};

const config = {
  initialColorMode: "light",
  useSystemColorMode: false,
};

const theme = extendTheme({ config });

const defaultDevice = deviceOptions[0];

const defaultData = {};

function PC() {
  const [device, setDevice] = useState(defaultDevice.value);

  const [graphOne, setGraphOne] = useState(defaultData);

  const [updatedGraphOne, setUpdatedGraphOne] = useState(graphOne);

  // const [activeCalories, setActiveCalories] = useState("");
  const [calories, setCalories] = useState("");
  const [steps, setSteps] = useState("");
  const [distance, setDistance] = useState("");
  const [sleep, setSleep] = useState("");

  //-------------progressive bar components------------------
  // const [activeCalories, setActiveCalories] = useState("");
  const [barCalories1, setBarCalories1] = useState("");
  const [barSteps1, setBarSteps1] = useState("");
  const [barDistance1, setBarDistance1] = useState("");
  const [barSleep1, setBarSleep1] = useState("");
  const [value5, setValue5] = useState("");
  const [value6, setValue6] = useState("");
  const [value7, setValue7] = useState("");

  const [barCalories2, setBarCalories2] = useState("");
  const [barSteps2, setBarSteps2] = useState("");
  const [barDistance2, setBarDistance2] = useState("");
  const [barSleep2, setBarSleep2] = useState("");

  const [barCalories3, setBarCalories3] = useState("");
  const [barSteps3, setBarSteps3] = useState("");
  const [barDistance3, setBarDistance3] = useState("");
  const [barSleep3, setBarSleep3] = useState("");

  const [barCalories4, setBarCalories4] = useState("");
  const [barSteps4, setBarSteps4] = useState("");
  const [barDistance4, setBarDistance4] = useState("");
  const [barSleep4, setBarSleep4] = useState("");

  const handleSum = () => {
    var sumCalories = graphOne.reduce(function (previousValue, currentValue) {
      return {
        totalCalories: previousValue.totalCalories + currentValue.totalCalories,
      };
    });

    var sumSteps = graphOne.reduce(function (previousValue, currentValue) {
      return {
        totalSteps: previousValue.totalSteps + currentValue.totalSteps,
      };
    });

    var sumDistance = graphOne.reduce(function (previousValue, currentValue) {
      return {
        totalDistance: previousValue.totalDistance + currentValue.totalDistance,
      };
    });

    var sumSleep = graphOne.reduce(function (previousValue, currentValue) {
      return {
        totalSleepTime:
          previousValue.totalSleepTime + currentValue.totalSleepTime,
      };
    });

    setCalories(Object.values(sumCalories));
    setSteps(Object.values(sumSteps));
    setDistance(Object.values(sumDistance));
    setSleep(Object.values(sumSleep));
  };

  const handleProgress = () => {
    const graphWeek1 = graphOne.slice(0, 7);
    const graphWeek2 = graphOne.slice(7, 14);
    const graphWeek3 = graphOne.slice(14, 21);
    const graphWeek4 = graphOne.slice(21, 28);
    console.log(graphOne);

    var sumCalories1 = graphWeek1.reduce(function (
      previousValue,
      currentValue
    ) {
      return {
        totalCalories: previousValue.totalCalories + currentValue.totalCalories,
      };
    });
    var sumSteps1 = graphWeek1.reduce(function (previousValue, currentValue) {
      return {
        totalSteps: previousValue.totalSteps + currentValue.totalSteps,
      };
    });
    var sumDistance1 = graphWeek1.reduce(function (
      previousValue,
      currentValue
    ) {
      return {
        totalDistance: previousValue.totalDistance + currentValue.totalDistance,
      };
    });
    var sumSleep1 = graphWeek1.reduce(function (previousValue, currentValue) {
      return {
        totalSleepTime:
          previousValue.totalSleepTime + currentValue.totalSleepTime,
      };
    });
    const value5 = 50;
    const value6 = 75;
    const value7 = 35;
    setBarCalories1(Object.values(sumCalories1));
    setBarSteps1(Object.values(sumSteps1));
    setBarDistance1(Object.values(sumDistance1));
    setBarSleep1(Object.values(sumSleep1));
    setValue5(value5);
    setValue6(value6);
    setValue7(value7);

    var sumCalories2 = graphWeek2.reduce(function (
      previousValue,
      currentValue
    ) {
      return {
        totalCalories: previousValue.totalCalories + currentValue.totalCalories,
      };
    });
    var sumSteps2 = graphWeek2.reduce(function (previousValue, currentValue) {
      return {
        totalSteps: previousValue.totalSteps + currentValue.totalSteps,
      };
    });
    var sumDistance2 = graphWeek2.reduce(function (
      previousValue,
      currentValue
    ) {
      return {
        totalDistance: previousValue.totalDistance + currentValue.totalDistance,
      };
    });
    var sumSleep2 = graphWeek2.reduce(function (previousValue, currentValue) {
      return {
        totalSleepTime:
          previousValue.totalSleepTime + currentValue.totalSleepTime,
      };
    });

    setBarCalories2(Object.values(sumCalories2));
    setBarSteps2(Object.values(sumSteps2));
    setBarDistance2(Object.values(sumDistance2));
    setBarSleep2(Object.values(sumSleep2));

    var sumCalories3 = graphWeek3.reduce(function (
      previousValue,
      currentValue
    ) {
      return {
        totalCalories: previousValue.totalCalories + currentValue.totalCalories,
      };
    });
    var sumSteps3 = graphWeek3.reduce(function (previousValue, currentValue) {
      return {
        totalSteps: previousValue.totalSteps + currentValue.totalSteps,
      };
    });
    var sumDistance3 = graphWeek3.reduce(function (
      previousValue,
      currentValue
    ) {
      return {
        totalDistance: previousValue.totalDistance + currentValue.totalDistance,
      };
    });
    var sumSleep3 = graphWeek3.reduce(function (previousValue, currentValue) {
      return {
        totalSleepTime:
          previousValue.totalSleepTime + currentValue.totalSleepTime,
      };
    });

    setBarCalories3(Object.values(sumCalories3));
    setBarSteps3(Object.values(sumSteps3));
    setBarDistance3(Object.values(sumDistance3));
    setBarSleep3(Object.values(sumSleep3));

    var sumCalories4 = graphWeek4.reduce(function (
      previousValue,
      currentValue
    ) {
      return {
        totalCalories: previousValue.totalCalories + currentValue.totalCalories,
      };
    });
    var sumSteps4 = graphWeek4.reduce(function (previousValue, currentValue) {
      return {
        totalSteps: previousValue.totalSteps + currentValue.totalSteps,
      };
    });
    var sumDistance4 = graphWeek4.reduce(function (
      previousValue,
      currentValue
    ) {
      return {
        totalDistance: previousValue.totalDistance + currentValue.totalDistance,
      };
    });
    var sumSleep4 = graphWeek4.reduce(function (previousValue, currentValue) {
      return {
        totalSleepTime:
          previousValue.totalSleepTime + currentValue.totalSleepTime,
      };
    });

    setBarCalories4(Object.values(sumCalories4));
    setBarSteps4(Object.values(sumSteps4));
    setBarDistance4(Object.values(sumDistance4));
    setBarSleep4(Object.values(sumSleep4));
  };

  const handleDeviceChange = (value) => {
    const device = value.value;
    setDevice(device);
    handleSum();
    handleProgress();
  };

  const handleTimeChange = (value) => {
    const time = value.value;
    if (time === "thisMonth") {
      setUpdatedGraphOne(graphOne);
    }
    console.log(updatedGraphOne);
  };

  useEffect(() => {
    getDeviceData(device).then((updatedGraphOne) => {
      setUpdatedGraphOne(updatedGraphOne);
    });
  }, [device]);

  useEffect(() => {
    getDeviceData(device).then((graphOne) => {
      setGraphOne(graphOne);
    });
  }, [device]);

  return (
    <ThemeProvider theme={theme}>
      <Flex
        // marginLeft={251}
        marginLeft={30}
        paddingLeft={55}
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
                      <Text fontSize={24} color="#5F6AC4"> </Text>
                    </Flex>
                  </Box>
                </Flex>
              </Flex>
            </Flex>
            <Flex>
              <Flex
                width="634px"
                height="308px"
                borderRadius={30}
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
              </Flex> */}
            </Flex>
          </Box>
          {/*
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
        <Box marginRight={50}>{/* <Text>Settings Logo</Text> */}
        <Box marginRight={50}></Box>
        </Box>
      </Flex> 
    </ThemeProvider>
  );
}

export default PC;