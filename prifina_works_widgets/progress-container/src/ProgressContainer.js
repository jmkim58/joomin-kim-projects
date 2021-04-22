import React, { useState } from "react";

import { Text, Flex } from "@blend-ui/core";

import {
  extendTheme,
  ThemeProvider,
  Progress,
  Stack,
  Tooltip,
  WrapItem,
} from "@chakra-ui/react";

const styles = {
  transform: `rotate(${270}deg)`,
};

const config = {
  initialColorMode: "light",
  useSystemColorMode: false,
};

const theme = extendTheme({ config });

function ProgressContainer({
  text,
  value1,
  value2,
  value3,
  value4,
  value5,
  value6,
  value7,
}) {
  return (
    <ThemeProvider theme={theme}>
      <Flex flexDirection="column">
        <div style={styles}>
          <Stack spacing="10px">
            <Tooltip
              label={value1 + " kcal"}
              aria-label="A tooltip"
              backgroundColor="yellow.200"
              textColor="#5F6AC4"
              closeDelay={100}
            >
              <span>
                <Progress
                  colorScheme="yellow"
                  size="sm"
                  value={value1 / 350}
                  width={150}
                  height="5px"
                  borderRadius={100}
                  backgroundColor="white"
                />
              </span>
            </Tooltip>
            <Tooltip
              label={value2 + " steps"}
              aria-label="A tooltip"
              backgroundColor="orange.100"
              textColor="#5F6AC4"
              closeDelay={100}
            >
              <span>
                <Progress
                  colorScheme="orange"
                  size="sm"
                  value={value2 / 1500}
                  width={150}
                  height="5px"
                  borderRadius={100}
                  backgroundColor="white"
                />
              </span>
            </Tooltip>
            <Tooltip
              label={(value3 / 1000).toFixed(1) + " km"}
              aria-label="A tooltip"
              backgroundColor="green.100"
              textColor="#5F6AC4"
              closeDelay={100}
            >
              <span>
                <Progress
                  colorScheme="green"
                  size="sm"
                  value={value3 / 1600}
                  width={150}
                  height="5px"
                  borderRadius={100}
                  backgroundColor="white"
                />
              </span>
            </Tooltip>
            <Tooltip
              label={(value4 / 60).toFixed(0) + " hrs"}
              aria-label="A tooltip"
              backgroundColor="blue.100"
              textColor="#5F6AC4"
              closeDelay={100}
            >
              <span>
                <Progress
                  colorScheme="blue"
                  size="sm"
                  value={value4 / 55}
                  width={150}
                  height="5px"
                  borderRadius={100}
                  backgroundColor="white"
                />
              </span>
            </Tooltip>
            <Progress
              colorScheme="purple"
              size="sm"
              value={value5}
              width={132}
              height="5px"
              borderRadius={100}
              backgroundColor="white"
            />
            <Progress
              colorScheme="pink"
              size="sm"
              value={value6}
              width={147}
              height="5px"
              borderRadius={100}
              backgroundColor="white"
            />
            <Progress
              colorScheme="cyan"
              size="sm"
              value={value7}
              width={109}
              height="5px"
              borderRadius={100}
              backgroundColor="white"
            />
          </Stack>
        </div>
        <div style={{ marginTop: 50, marginLeft: 53 }}>
          <Text color="#95A4B7" fontSize={15}>
            {text}
          </Text>
        </div>
      </Flex>
    </ThemeProvider>
  );
}

export default ProgressContainer;

{
  /* <div style={styles}>
        <div style={{ padding: 5, width: 100 }}>
          <LinearProgress variant="determinate" value={0} />
        </div>
        <div style={{ padding: 5, width: 132 }}>
          <LinearProgress variant="determinate" value={0} />
        </div>
        <div style={{ padding: 5, width: 147 }}>
          <LinearProgress variant="determinate" value={0} />
        </div>
        <div style={{ padding: 5, width: 109 }}>
          <LinearProgress variant="determinate" value={0} />
        </div>
        <div style={{ padding: 5, width: 132 }}>
          <LinearProgress variant="determinate" value={100} />
        </div>
        <div style={{ padding: 5, width: 147 }}>
          <LinearProgress variant="determinate" value={100} color="secondary" />
        </div>
        <div style={{ padding: 5, width: 109 }}>
          <LinearProgress variant="determinate" value={100} color="primary" />
        </div>
      </div> */
}
