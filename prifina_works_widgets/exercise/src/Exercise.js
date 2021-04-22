import React, { useState, useEffect } from "react";

//import { Text, Flex } from "@blend-ui/core";
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

import {Text, Flex, extendTheme, ThemeProvider, Grid, GridItem} from "@chakra-ui/react";
import WeatherMain from "../src/components/WeatherMain";
import {ReactComponent as RunIcon} from "../src/assets/run.svg";


// -----------------------Config----------------------

const config = {
    initialColorMode: "light",
    useSystemColorMode: false,
};

const theme = extendTheme({ config });


// ----------------Style-----------------------

const styles = {
    boxShadow: " 0px 5px 20px #F0F4F8",
};

//---------------DATA------------------------

let dataNew = [
  { name: "March 20", startTime: 17 },
  { name: "March 21", startTime: 16 },
  { name: "March 22", startTime: 16 },
  { name: "March 23", startTime: 15 },
  { name: "March 24", startTime: 3 },
  { name: "March 25", startTime: 14 },
  { name: "March 26", startTime: 17 },
  { name: "March 27", startTime: 7 },
  { name: "March 28", startTime: 16 },
  { name: "March 29", startTime: 15 },

];

let startTimeList = dataNew.map((run) => run.startTime);

//--------------APP FUNCTION--------------------------

function Exercise() {

    //------------------Constants-------------------------

    const [average, setAverage] = useState('16:24');
    const [lowerBound, setLowerBound] = useState('16:00')
    const [upperBound, setUpperBound] = useState('18:00')
    const [runCount, setRunCount] = useState('5')

    let iconStyles = { color: "white"};

    //--------------------FUNCTIONS------------------------------
      
    //METHOD: Function to caluclate the average of an array (argument = arr) for 10 items
    const getAverage = arr =>{
        //Sum the values in the array
        const reducer = (total, currentValue) => total + currentValue;
        const sum = dataNew.reduce(reducer);
        console.log(sum/10);

        //divide the array sum by the length 10
        return sum / 10;
        //const u = 10;
        //return u;

      }

    //METHOD: Returns the mode of an array 
    const getMode = arr =>{
        const obj ={};

        //Count the number of occurences for each object in the array
        arr.forEach(number => {
          if (!obj[number]){
            obj[number] = 1;
          }
          else {
            obj[number] += 1;
          }
        });

        //Obj: '3':2, '4':4, '10': 3 ...etc

        let highestValue = 0;
        let highestValueKey = 1000;

        //Loop over object and find object with highest value
        for (let key in obj){
          const value = obj[key];
          if (value > highestValue){
            highestValue = value;
            highestValueKey = key;
          }
        }

        //Convert the string keys into a number value
        setLowerBound(Number(highestValueKey) - 1);
        setUpperBound(Number(highestValueKey) + 1);
        console.log(lowerBound);
        console.log(upperBound);
      }

    //METHOD: Counts the number of times a value in the arr occurs within the lower and upper bounds
    const getRunCount = function(arr, val, val2){

        //Iterate over each element using reduce and add to count if element is between upper and lowerbounds
        const runCount = arr.reduce((count, element) => {
          return ((element <= val && element >= val2) ? count + 1 : count)
        }, 0);


        setRunCount(runCount);
      };


    //--------------------useEffect()--------------------------

    useEffect(() => {
      console.log(startTimeList); //startTime Array 
      getMode(startTimeList);  //Find Mode of StartTime Array and finds the lower and upperbounds

      getRunCount(startTimeList, upperBound, lowerBound); //Calculate run count within boundaries
    })

    return (
        <ThemeProvider theme={theme}>

          <Box style={styles} width={600} height={270} borderRadius={10}>
            <Flex 
                bg={"#14233d"}   //Added Background for Widget
                borderRadius = {20} //Rounded border radius 
            > 
              <Box style={styles} width={600} height={260} borderRadius={10}>
                <Flex
                  justifyContent={"space-between"}
                  alignItems={"center"}
                  paddingLeft={30}
                  paddingRight={30}
                  paddingTop={20}
                  >
                  <Flex>
                    <Text
                      fontStyle="normal"
                      fontFamily= 'Slab Serifs'
                      style={{
                        textDecoration: "none",
                        fontWeight: 900,
                        color: "#db3e00",
                        fontSize: 24,
                        }}
                    >
                        Exercise
                    </Text>
                  </Flex>

                </Flex>
                <Grid
                    h="200px"
                    templateRows="repeat(10, 1fr)"
                    templateColumns="repeat(5, 1fr)"
                    gap={2}
                    >
                    <GridItem rowStart={1} rowEnd={10} colStart={1} colEnd={3}>
                        <Grid
                            h="200px"
                            templateRows="repeat(2, 1fr)"
                            templateColumns="repeat(1, 1fr)"
                            gap={2}>
                                <GridItem rowSpan={1} >
                                    <Flex
                                        marginRight={10}
                                        marginTop = {40}
                                        marginLeft={18}>
                                        <Flex
                                            marginRight={12}
                                            marginLeft = {0}
                                            >
                                            
                                            <RunIcon size="2.2em" color="white" stroke = "white" fill="white" style={iconStyles}> </RunIcon>
                                            
                                             
                                        </Flex>
                                        <Flex
                                            marginLeft={-2}
                                            justifyContent={"space-between"}
                                            flexDirection={"column"}>

                                              <Text
                                                style={{
                                                  textDecoration: "none",
                                                  fontWeight: 400,
                                                  color: "#FFFFFF",
                                                  fontSize: 21,
                                                  }}>
                                                Planning to run?
                                                </Text>

                                                <div style={{ 
                                                  textDecoration: "none",
                                                  fontWeight: 400,
                                                  color: "#7fffd4",
                                                  fontSize: 18.5,}}>

                                                {lowerBound + ":00"} to {upperBound + ":00"}   
                                              </div>
                                            </Flex>
                                    </Flex>
                                    
                                </GridItem>

                                <GridItem rowSpan={1}>
                                        <WeatherMain></WeatherMain>                           
                                </GridItem>

                        </Grid>
                  
                    </GridItem>
                    <GridItem rowStart={1} rowEnd={4} colStart={3} colEnd={6}>
                        <Flex
                            justifyContent={"space-between"}
                            alignItems={"center"}
                            paddingLeft={0}
                            paddingRight={20}
                            paddingTop={0}
                            >
                            <Text
                                fontStyle="normal"
                                style={{
                                  textDecoration: "none",
                                  fontWeight: 400,
                                  color: "#FFFFFF",
                                  fontSize: 20,
                                  }}
                                >
                                You have run between {lowerBound + ":00"} and {upperBound + ":00"} {runCount} out of your last 10 runs.
                            </Text>
                        </Flex>
                    
                    </GridItem> 
                    <GridItem rowStart={4} rowEnd={11} colStart={3} colEnd={6}>

                         <ResponsiveContainer>
                            <AreaChart
                                datakey = "startTime"
                                data = {dataNew}
                                width = {250}
                                height={250}
                                margin={{
                                top: 12,
                                right: 50,
                                left: -30,
                                bottom: -10,
                                }}
                            >
                                <CartesianGrid
                                strokeDasharray="3 3"
                                horizontal={false}
                                vertical={false}
                                />
                                <XAxis dataKey="name" tick={false} axisLine={false} />
                                <YAxis tick={true} axisLine={false} stroke="#E9EDF0"/> {/*Changed color of Y-axis Tick */}
                                <Tooltip position={{ y: 95 }} fill={"red"}/>
                                <defs>
                                <linearGradient
                                    id="colorUv"
                                    x1="0"
                                    y1="0 "
                                    x2="0"
                                    y2="100%"
                                    spreadMethod="reflect"
                                >
                                    <stop offset="0" stopColor="#FFD46A" />
                                    <stop offset="1" stopColor="white" /> 
                                </linearGradient>  
                                </defs>
                                <Area
                                type ="monotone"
                                dataKey ="startTime"
                                data = {dataNew}
                                stroke="orange" 
                                fill="url(#colorUv)"
                                />
                             </AreaChart>
                        </ResponsiveContainer>
                    </GridItem>
                </Grid>                
              </Box>
            </Flex>
          </Box>
        </ThemeProvider>

    );

}

export default Exercise;