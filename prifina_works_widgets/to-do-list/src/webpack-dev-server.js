/**
 * webpack-dev-server entry point for debugging.
 * This file is not bundled with the library during the build process.
 */
import React from "react";
import ReactDOM from "react-dom";

// import { LocalComponent } from "./App.js";
import { App } from "./App.js"

const node = document.getElementById("app");

// const App = (props) => <LocalComponent {...props} />;
// const App = (props) => <PC />;


ReactDOM.render(<App msg={"Hello, Tero"} />, node);

{/*
ReactDOM.render (
    <div>
        <ToDoList />
    </div>,
    destination
);
*/}