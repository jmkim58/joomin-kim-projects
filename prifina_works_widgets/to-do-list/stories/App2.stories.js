import React, { useState, useEffect } from "react";
// import App from "../src/Hello";
import App2 from "../src/App2";
import CheckBox from "../src/CheckBox"

export default { title: "Test" };

export const app2 = () => {
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
  return <App2 />;
};

app2.story = {
  name: "App2",
};
