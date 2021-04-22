import React, { useState, useEffect, Component } from "react";
import ReactDOM from 'react-dom';

import { Text, Flex } from "@blend-ui/core";
import Select from "react-select";

import Box from "@blend-ui/core/dist/esm/Box";

import ProgressContainer from "../src/ProgressContainer";
import ProgressCard from "../src/ProgressCard";
import Card from "../src/Card";

import { extendTheme, ThemeProvider } from "@chakra-ui/react";

import { deviceOptions, getDeviceData } from "../src/data/helper";

import { selectStyleShadow } from "../src/styles/styles";
import './styles/styles';

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

const toDoCheck = [{ title: 'item1', isDone: false }, 
                   { title: 'item2', isDone: true }]


{/*
class TDL extends Component {
    render() {
        
    }
}
*/}



function ToDoList() {

    const toDoCheck = [{ title: 'item1', isDone: false }, 
    { title: 'item2', isDone: true }]

    const listItems = [
        "Walk the dog", "Buy groceries", "Call mom"
    ]

    const [ items, setItem ] = useState(listItems);


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
                {/*<Text fontSize={34} color="#5F6AC4">
                  // Health Overview
                // </Text>*/}
                    
                    <h1 style={{fontFamily: "Fredoka One"}}> TO-DO LIST </h1>
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
            <Flex>
              <Flex
                width="634px"
                height="308px"
                borderRadius={30}
                paddingLeft={50}
                paddingTop={20}
                style={styles}
                justifyContent="left"
                alignItems="left"
              >

                {/*
                <div>
                    <h1 style={{fontFamily: "Dawning of a New Day", fontSize: 30}}> a list of things to do! </h1>
                    {items.map((item) => {
                        return <div key={item}>{item}</div>;
                    })}
                </div>
                */}


                <div className="toDoListMain">
                    <div className="header">
                    <p style={{fontFamily: "Fredoka One", fontSize: 20}}> Create your own To-Do List </p>
                        <form>
                            <input placeholder="Enter task: ">
                            </input>
                            <button type="submit">Add</button>
                        </form>
                    </div>
                </div>


             </Flex>
            </Flex>
          </Box>
        </Box>
     </Flex>
    </ThemeProvider>
  );
}


function TodoItem({ title, isDone }) {
    return (
      <div>
        <p style={{ color: isDone? 'green': 'red' }}>{title}</p>
      </div>
  )
   }

function ToDo(props) {
    const toDoCheck = props.toDoCheck.map(el => <ToDoItem key={el.id} title={el.title} isDone={el.isDone} />)
    return <div>{toDoCheck}</div>
}



// ReactDOM.render(<ToDo toDoCheck={todoCheck} />, document.getElementById('app'));

export default ToDoList;