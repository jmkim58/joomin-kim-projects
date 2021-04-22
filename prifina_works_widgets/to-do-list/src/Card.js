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

function Card({
  icon,
  smallIcon,
  precentage,
  title,
  subtitle,
  iconColor,
  color,
}) {
  return (
    <ThemeProvider theme={theme}>
      <Flex
        flex={1}
        width="185px"
        height="118px"
        borderRadius={10}
        style={styles}
        paddingLeft={20}
        paddingRight={20}
        paddingBottom={20}
        paddingTop={5}
        bg={color}
      >
        <Box>
          <Flex paddingBottom={10}>
            <img src={smallIcon} />
            <Text fontSize="12px" color="#5F6AC4" paddingLeft={5}>
              {precentage}
            </Text>
          </Flex>
          <Flex
            borderRadius="9999px"
            bg={iconColor}
            width="52px"
            height="52px"
            alignItems="center"
            justifyContent="center"
          >
            <img src={icon} />
          </Flex>
        </Box>
        <Box marginLeft={16}>
          <Box marginBottom={5} marginTop={35}>
            <Text color="#5F6AC4" fontSize={18}>
              {title}
            </Text>
          </Box>
          <Box>
            <Text color="#5F6AC4" fontSize={14}>
              {subtitle}
            </Text>
          </Box>
        </Box>
      </Flex>
    </ThemeProvider>
  );
}

export default Card;
