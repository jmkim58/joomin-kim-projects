import React from "react";

import { Text, Flex } from "@blend-ui/core";

import Box from "@blend-ui/core/dist/esm/Box";

import {
  extendTheme,
  ThemeProvider,
  CircularProgress,
  CircularProgressLabel,
} from "@chakra-ui/react";

const styles = {
  boxShadow: " 0px 5px 20px #F0F4F8",
};

const config = {
  initialColorMode: "light",
  useSystemColorMode: false,
};

const theme = extendTheme({ config });

function ProgressCard({ title, subtitle, color, value, icon }) {
  return (
    <ThemeProvider theme={theme}>
      <Flex
        flex={1}
        width="353px"
        height="137px"
        borderRadius={10}
        style={styles}
        padding={20}
      >
        <CircularProgress
          value={value}
          color={color}
          thickness="11px"
          size="95px"
          trackColor="transparent"
        >
          <CircularProgressLabel>
            <img src={icon} />
          </CircularProgressLabel>
        </CircularProgress>
        <Box marginLeft={35}>
          <Box marginBottom={5} marginTop={27}>
            <Text color="#5F6AC4" fontSize={28}>
              {title}
            </Text>
          </Box>
          <Box>
            <Text color="#5F6AC4" fontSize={18}>
              {subtitle}
            </Text>
          </Box>
        </Box>
      </Flex>
    </ThemeProvider>
  );
}

export default ProgressCard;
