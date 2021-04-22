import React, { Component } from "react";
import { extendTheme, ThemeProvider, Checkbox, CheckboxGroup} from "@chakra-ui/react";

class TodoItems extends Component {
  createTasks(item) {
    return <li key={item.key}>{item.text}</li>
  }


 
  render() {
    var todoEntries = this.props.entries;
    var listItems = todoEntries.map(this.createTasks);
 
    return (
      <ul className="theList" style={{fontFamily: "Poppins", fontSize: 15}}>
          {listItems}
      </ul>
    );
  }
};
 
export default TodoItems;