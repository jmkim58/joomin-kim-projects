import React, { Component } from 'react';
import CheckBox from '../src/CheckBox';

// import { ThemeProvider } from "@blend-ui/core";

import { Text, Flex } from "@blend-ui/core";
import Select from "react-select";
// import '../src/styles/styles';

import { Box, BoxGroup, Container } from "@chakra-ui/react"
// import Box from "@blend-ui/core/dist/esm/Box";

import { extendTheme, theme, ThemeProvider, Checkbox, 
         CheckboxGroup, Button, ButtonGroup, Wrap, WrapItem, Switch, Form, Field,
         FormControl,
         FormLabel,
         FormErrorMessage,
         FormHelperText } from "@chakra-ui/react";

import { CheckCircleIcon } from '@chakra-ui/icons'

class App2 extends Component {
  constructor(props) {
    super(props)
    this.state = {
      google_data_items: [
        {id: 1, value: "Google Data1", isChecked: false},
        {id: 2, value: "Google Data2", isChecked: false},
        {id: 3, value: "Google Data3", isChecked: false},
        {id: 4, value: "Google Data4", isChecked: false}
      ]
    }
  }

  handleAllChecked = (event) => {
      let google_data_items = this.state.google_data_items
      google_data_items.forEach(google_data_item => google_data_item.isChecked = event.target.checked)
      this.setState({google_data_items: google_data_items})
  }

  handleCheckElement = (event) => {
      let google_data_items = this.state.google_data_items
      google_data_items.forEach(google_data_item => {
          if (google_data_item.value === event.target.value)
          google_data_item.isChecked = event.target.checked
      })
      this.setState({google_data_items: google_data_items})
  }
  
  render() {
    return (
      <div className="App2">

    


     <ThemeProvider theme={theme}>
      <Flex
        // marginLeft={251}
        marginLeft={30}
        paddingLeft={55}
        paddingTop={20}
        // justifyContent={"space-between"}
      >
        <Box>
          <Box marginBottom={75} paddingTop={0}>
            <Flex justifyContent={"space-between"}>
              <Flex marginBottom={0}>
                <Box bg="#E9D8FD" w="100%" height="30px"
                    paddingTop={20} paddingBottom={17} paddingLeft={30} paddingRight={30} borderRadius={30}
                    fontFamily="Poppins" color="#285E61" marginBottom={20}>
                {/*<Text fontSize={34} color="#5F6AC4">
                  // Health Overview
                // </Text>*/}
                    <Text fontSize={34} style={{fontFamily: "Fredoka One"}}> <CheckCircleIcon h="28px"/> To-Do List </Text> 
                    </Box>
              </Flex>
              {/*
              <Flex paddingRight={35}>
                <Flex>
                  <Box>
                    <Flex minHeight="35px">
                      <div
                        style={{
                          minWidth: 180,
                          paddingRight: 35,
                        }}
                      >
                          
                      </div>
                    </Flex>
                    <Flex paddingRight={300} paddingBottom={27}>
                      <Text fontSize={24} color="#5F6AC4">  </Text>
                    </Flex>
                  </Box>
                    </Flex>
                    </Flex> */}
            </Flex>
              <Flex
                width="500px"
                height="308px"
                borderRadius={30}
                paddingLeft={50}
                paddingTop={20}
                //style={styles}
                justifyContent="flex-start"
                alignItems="left"
                flexDirection="column"
                backgroundColor= "#E6FFFA" // "#F7FAFC"
                color="#2A4365" // "#A0AEC0"
              >
                <Wrap >
                <WrapItem>
                <Box bg="#B2F5EA" w="100%" height="30px"
                paddingRight={18} paddingTop={10} paddingLeft={15} borderRadius={20}
                fontFamily="Poppins" color="#285E61" marginTop={5} textAlign="justify">
                    <input type="checkbox" onChange={this.handleAllChecked}  value="checkedall" style={{fontFamily: "Poppins", fontWeight: 400, fontSize: 15}}/> Check / Uncheck All
                </Box> 
                </WrapItem>
                <Container flexDirection="column" paddingTop={10} paddingBottom={10} paddingLeft={0}>

                    <ul style={{fontFamily: "Poppins", fontWeight: 400, fontSize: 15}}>
                        {
                            this.state.google_data_items.map((google_data_item) => {
                                return (<CheckBox handleCheckElement={this.handleCheckElement} {...google_data_item} />)
                            })
                        }
                    </ul>
                </Container>
                <FormControl display="flex" alignItems="center">
                    <FormLabel htmlFor="email-alerts" mb="0" fontFamily="Poppins">
                        Do you want an e-mail reminder? 
                    </FormLabel>
                    <Switch id="email-alerts"/>
                    </FormControl>
                </Wrap>


 
      </Flex>
      </Box>
      </Box>
      </Flex>
      </ThemeProvider>
      </div>
    );
  }
}

export default App2