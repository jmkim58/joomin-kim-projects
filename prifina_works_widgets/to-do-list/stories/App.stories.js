import React, { useState, useEffect } from "react";
// import App from "../src/Hello";
import App from "../src/App";

export default { title: "Test" };

export const app = () => {
  /*
  const [onUpdate, set] = useState({});
  useEffect(() => {
    let timer = null;

    timer = setTimeout(() => {
      set({ msg: "OK" });
    }, 5000);

    return () => clearTimeout(timer);
  }, []);
  */

  // return <App msg={"Bye, Tero"} />;
  return <App />;
};

app.story = {
  name: "App",
};
