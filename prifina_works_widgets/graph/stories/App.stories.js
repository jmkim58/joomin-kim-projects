import React, { useState, useEffect } from "react";
import App from "../src/graph";

export default { title: "Graph" };

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

  return <App />;
};
app.story = {
  name: "App",
};
