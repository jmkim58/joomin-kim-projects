import React, { useState, useEffect } from "react";

import { Text, Flex } from "@blend-ui/core";  
import Box from "@blend-ui/core/dist/esm/Box";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  AreaChart,
  Area,
  ResponsiveContainer,
} from "recharts"; 

import { extendTheme, ThemeProvider } from "@chakra-ui/react"; 
import Select from "react-select"; 

import {
  timeOptions,
  deviceOptions,
  dataOptions,
  getDeviceData,
  getDeviceDataTwo,
  customTheme,
} from "../src/data/helper";

import {
  selectStyle,
  selectStyleOutline,
  selectStyleSmall,
} from "../src/styles/styles";

// -----------------------FUNCTIONS----------------------

const config = {
  initialColorMode: "light",
  useSystemColorMode: false,
};

//function for getting the sum of object arrays with string values
function sumProperty(arr, type) {
  return arr.reduce((total, obj) => {
    if (typeof obj[type] === "string") {
      return total + Number(obj[type]);
    }
    return total + obj[type];
  }, 0);
}

const theme = extendTheme({ config });

// -------------DATA----------------

const styles = {
  boxShadow: " 0px 5px 20px #F0F4F8",
};

const defaultDevice = deviceOptions[0]; //Fitbit 

const defaultData = {};

// --------------------------APP FUNCTION--------------------------------

function Graph() {
  const [device, setDevice] = useState(defaultDevice.value);
  const [deviceTwo, setDeviceTwo] = useState(defaultDevice.value);

  const [graphOne, setGraphOne] = useState(defaultData);
  const [graphTwo, setGraphTwo] = useState(defaultData);

  const [updatedGraphOne, setUpdatedGraphOne] = useState(graphOne);
  const [updatedGraphTwo, setUpdatedGraphTwo] = useState(graphTwo);

  const [dataType, updateDataType] = useState([
    "totalCalories",
    "activeCalories",
    "totalSteps",
    "totalDistance",
    "inactiveMinutes",
    "lowActiveMinutes",
    "mediumActiveMinutes",
    "highActiveMinutes",
    "lightSleepTime",
    "deepSleepTime",
    "timeSpentInBed",
    "totalSleepTime",
    "REMSleepTime",
    "totalAwakeTime",
    "restlessSleep",
    "averageHRV",
    "averageRestingHR",
    "respiratoryRate",
    "temperatureDeviation",
  ]);


  const [time, updateTime] = useState(["thisWeek", "lastWeek", "thisMonth"]);
  const [timeTwo, updateTimeTwo] = useState([
    "thisWeek",
    "lastWeek",
    "thisMonth",
  ]);

  const [sum, setSum] = useState("");

  const handleDeviceChange = (value) => {
    const device = value.value;
    setDevice(device);
  };

  const handleDataChange = (value) => {
    const number = value.value;
    // const array = graphOne;
    updateDataType(number);
    // console.log(number);
    // console.log(graphOne);

    // let totalAmount = sumProperty(array, dataType).toFixed(0);
    // console.log(totalAmount);
    // setSum(totalAmount);
  };


  const handleTimeChange = (value) => {
    const time = value.value;
    if (time === "thisWeek") {
      const array = graphOne.slice(21, 28);
      setUpdatedGraphOne(array);
    } else if (time === "lastWeek") {
      const array = graphOne.slice(14, 21);
      setUpdatedGraphOne(array);
    } else {
      setUpdatedGraphOne(graphOne);
    }
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

  useEffect(() => {
    getDeviceDataTwo(deviceTwo).then((updatedGraphTwo) => {
      setUpdatedGraphTwo(updatedGraphTwo);
    });
  }, [deviceTwo]);

  useEffect(() => {
    getDeviceDataTwo(deviceTwo).then((graphTwo) => {
      setGraphTwo(graphTwo);
    });
  }, [deviceTwo]);

  return (
    <ThemeProvider theme={theme}>
      <Flex
        marginLeft={2}
        justifyContent={"space-between"}
        flexDirection={"column"}
      >
        <Flex
          paddingLeft={70}
          paddingRight={140}
          justifyContent={"space-between"}
        >
              {/*Widget Number 1 - Fitbit Default*/}
          <Box>
            <Flex marginTop={40}>
              <Box style={styles} width={350} height={230} borderRadius={10}>
                <Flex
                  justifyContent={"space-between"}
                  alignItems={"center"}
                  paddingLeft={10}
                  paddingRight={30}
                  paddingTop={20}
                >
                  <Flex>
                    <div style={{ minWidth: 180 }}>
                      <Select
                        maxMenuHeight ={150}
                        styles={selectStyle}
                        options={dataOptions}
                        defaultValue="Data"
                        placeholder="Data"
                        onChange={handleDataChange}
                        width="100px"
                      />
                    </div>
                  </Flex>
                  <Flex>
                    <Text
                      color={"#5F6AC4"}
                      fontSize={28}
                      fontFamily="Circular Std"
                      fontWeight={900}
                      fontStyle="normal"
                    >
                      {/* {sum} */}
                      Fitbit
                    </Text>
                  </Flex>
                </Flex>
                <Flex
                  justifyContent={"space-between"}
                  alignItems={"center"}
                  paddingLeft={30}
                  paddingRight={30}
                >
                  <Flex>
                    <Flex>
                      <div style={{ minWidth: 130 }}>
                        <Select
                          maxMenuHeight ={120}
                          options={timeOptions}
                          styles={selectStyleSmall}
                          defaultValue="Time"
                          placeholder="Time"
                          onChange={handleTimeChange}
                          width="100px"
                        />
                      </div>
                    </Flex>
                  </Flex>
                  <Flex>
                    <div style={{ minWidth: 100 }}>
                      {/*<Select
                        options={deviceOptions}
                        styles={selectStyleOutline}
                        defaultValue="Device"
                        placeholder="Device"
                        onChange={handleDeviceChange}
                        width="100px"
                      /> 
                      */}
                    </div>
                  </Flex>
                </Flex>
                <ResponsiveContainer>
                  <AreaChart
                    data={updatedGraphOne}
                    syncId="anyId"
                    margin={{
                      top: 0,
                      right: 0,
                      left: -60,
                      bottom: 75,
                    }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      horizontal={false}
                      vertical={false}
                    />
                    <XAxis dataKey="name" tick={false} axisLine={false} />
                    <YAxis tick={false} axisLine={false} />
                    <Tooltip position={{ y: 90 }}/>
                    <defs>
                      <linearGradient
                        id="colorUv2"
                        x1="0%"
                        y1="0%"
                        x2="0%"
                        y2="100%"
                        spreadMethod="reflect"
                      >
                        <stop offset="0" stopColor="#5F6AC4" />
                        <stop offset="1" stopColor="white" />
                      </linearGradient>
                    </defs>
                    <Area
                      type="monotone"
                      dataKey={dataType}
                      stroke="blue"
                      fill="url(#colorUv2)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </Flex>
          </Box>
        </Flex>
      </Flex>
    </ThemeProvider>
  );
}

export default Graph;
